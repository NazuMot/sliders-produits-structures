"""
Version anglaise de l'appli, sans dupliquer le code.

Les textes affiches (chaines Python) sont traduits A LA VOLEE : en anglais,
chaque module du projet est relu, ses chaines sont remplacees par leur
traduction (traductions/en.json), puis il est execute sous un autre nom
(en_vues.brc, en_graphiques...). Le code francais reste la seule source.

Ce qui n'est pas traduit : docstrings, cles de widgets (key=...), identifiants
(chaines en minuscules sans espace), couleurs, champs Altair ("serie:N").
Une chaine absente du dictionnaire reste en francais.

f-strings : la cle est le gabarit, chaque {expression} devient {0}, {1}...
Pour lister les textes a traduire : python scripts/traductions.py
"""
import ast
import importlib.abc
import importlib.util
import json
import os
import re
import sys

RACINE = os.path.dirname(os.path.abspath(__file__))
TRADUITS = {"principal", "barre", "graphiques", "sales", "backtest", "vues"}
IDENT = re.compile(r"^[a-z0-9_]*$")
CODE = re.compile(r"^(#[0-9A-Fa-f]{3,8}|[A-Za-z_][\w]*:[QNOT]|.*datum\..*)$")
KW_IGNORES = {"key", "bind", "type", "anchor", "orient", "interpolate", "baseline", "align", "on", "clear",
              "selection_mode", "on_select", "width", "label_visibility", "fontWeight", "symbolType",
              "labelExpr", "theme", "creance", "marge_sur", "format", "mode", "fields", "name"}
_TABLE = None


def table():
    global _TABLE
    if _TABLE is None:
        p = os.path.join(RACINE, "traductions", "en.json")
        _TABLE = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    return _TABLE


class _TraduitConstantes(ast.NodeTransformer):
    """Traduit les textes cites dans une expression : {'continue' if c else 'à l'échéance'}."""

    def __init__(self, tab):
        self.tab = tab

    def visit_Constant(self, n):
        if isinstance(n.value, str) and n.value in self.tab and not IDENT.match(n.value):
            return ast.copy_location(ast.Constant(self.tab[n.value]), n)
        return n


def _champ(fv, tab=None):
    """Source d'un champ {expr!r:spec} d'une f-string (textes cites traduits si tab)."""
    import copy
    if tab:
        fv = _TraduitConstantes(tab).visit(copy.deepcopy(fv))
    s = ast.unparse(ast.JoinedStr(values=[fv]))
    return s[2:-1]


def cibles(src):
    """(noeud, cle) des chaines affichees d'un source."""
    t = ast.parse(src)
    ign = set()
    for n in ast.walk(t):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body \
           and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, (ast.Constant, ast.JoinedStr)):
            ign.add(id(n.body[0].value))
        if isinstance(n, ast.keyword) and n.arg in KW_IGNORES:
            for c in ast.walk(n.value):
                ign.add(id(c))
        if isinstance(n, ast.JoinedStr):
            for v in n.values:
                for c in ast.walk(v):
                    ign.add(id(c))
    out = []
    for n in ast.walk(t):
        if id(n) in ign:
            continue
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            if IDENT.match(n.value) or CODE.match(n.value) or not re.search(r"[A-Za-zÀ-ÿ]", n.value):
                continue
            out.append((n, n.value))
        elif isinstance(n, ast.JoinedStr):
            k, champs = "", []
            for v in n.values:
                if isinstance(v, ast.Constant):
                    k += v.value.replace("{", "{{").replace("}", "}}")
                else:
                    k += "{%d}" % len(champs)
                    champs.append(v)
            if re.search(r"[A-Za-zÀ-ÿ]", re.sub(r"\{\d+\}", "", k)):
                out.append((n, k, champs))
    return out


def _litteral_f(gabarit, champs):
    corps = gabarit
    for i, c in enumerate(champs):
        corps = corps.replace("{%d}" % i, "\x00%d\x00" % i)
    corps = corps.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r")
    q = '"""' if '"""' not in "".join(champs) and not corps.endswith('"') else "'''"
    for i, c in enumerate(champs):
        corps = corps.replace("\x00%d\x00" % i, c)
    return "f" + q + corps + q


def traduit_source(src, tab):
    lignes = src.split("\n")
    debut = [0]
    for l in lignes:
        debut.append(debut[-1] + len(l) + 1)

    def pos(ligne, col):
        return debut[ligne - 1] + len(lignes[ligne - 1].encode()[:col].decode())

    rempl = []
    for c in cibles(src):
        n, cle = c[0], c[1]
        en = tab.get(cle)
        if not en or en == cle:
            continue
        nouveau = repr(en) if len(c) == 2 else _litteral_f(en, [_champ(v, tab) for v in c[2]])
        rempl.append((pos(n.lineno, n.col_offset), pos(n.end_lineno, n.end_col_offset), nouveau))
    for a, b, r in sorted(rempl, reverse=True):
        src = src[:a] + r + src[b:]
    # les modules du projet importes depuis un module traduit sont eux aussi traduits
    src = re.sub(r"^(\s*)(from|import)\s+(principal|barre|graphiques|sales|backtest|vues)\b",
                 r"\1\2 en_\3", src, flags=re.M)
    return src


class _Chargeur(importlib.abc.Loader):
    def __init__(self, chemin, paquet):
        self.chemin, self.paquet = chemin, paquet

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        if self.paquet:
            module.__path__ = []
            return
        module.__file__ = self.chemin
        src = open(self.chemin, encoding="utf-8").read()
        code = compile(traduit_source(src, table()), self.chemin, "exec")
        exec(code, module.__dict__)


class _Chercheur(importlib.abc.MetaPathFinder):
    def find_spec(self, nom, path=None, target=None):
        if not nom.startswith("en_"):
            return None
        reel = nom[3:]
        if reel.split(".")[0] not in TRADUITS:
            return None
        if reel == "vues":
            return importlib.util.spec_from_loader(nom, _Chargeur(None, True), is_package=True)
        chemin = os.path.join(RACINE, *reel.split(".")) + ".py"
        return importlib.util.spec_from_loader(nom, _Chargeur(chemin, False))


if not any(isinstance(f, _Chercheur) for f in sys.meta_path):
    sys.meta_path.insert(0, _Chercheur())


def module(nom, anglais):
    """Le module du projet, en francais ou traduit en anglais."""
    return importlib.import_module(("en_" if anglais else "") + nom)


def bascule():
    """
    Changement de langue : les reglages texte (page choisie, options des menus)
    sont traduits pour rester valides dans l'autre langue.
    """
    import streamlit as st
    anglais = st.session_state.get("anglais", False)
    tab = table()
    sens = tab if anglais else {v: k for k, v in tab.items()}
    for cle in list(st.session_state.keys()):
        v = st.session_state[cle]
        if cle != "anglais" and isinstance(v, str) and v in sens:
            st.session_state[cle] = sens[v]
