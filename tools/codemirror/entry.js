// CodeMirror 6 bundle for the workbook → static/vendor/codemirror/codemirror.min.js (IIFE, works from file://).
// Rebuild: cd tools/codemirror && npm install && npm run build
import { EditorState } from "@codemirror/state";
import {
  EditorView, keymap, lineNumbers, highlightActiveLine, highlightActiveLineGutter,
  drawSelection, highlightSpecialChars,
} from "@codemirror/view";
import { defaultKeymap, history, historyKeymap, indentWithTab } from "@codemirror/commands";
import { indentOnInput, bracketMatching, syntaxHighlighting, HighlightStyle, indentUnit } from "@codemirror/language";
import { closeBrackets, closeBracketsKeymap } from "@codemirror/autocomplete";
import { python } from "@codemirror/lang-python";
import { tags as t } from "@lezer/highlight";

// Colours are CSS variables from tokens.css, so the editor follows the light/dark theme.
const highlight = HighlightStyle.define([
  { tag: [t.keyword, t.controlKeyword, t.definitionKeyword, t.moduleKeyword, t.operatorKeyword], color: "var(--syn-keyword)" },
  { tag: [t.function(t.definition(t.variableName)), t.function(t.variableName)], color: "var(--syn-func)" },
  { tag: t.definition(t.variableName), color: "var(--syn-name)" },
  { tag: [t.className, t.definition(t.className)], color: "var(--syn-entity)" },
  { tag: [t.string, t.special(t.string)], color: "var(--syn-string)" },
  { tag: [t.number, t.bool, t.null], color: "var(--syn-number)" },
  { tag: [t.operator, t.compareOperator, t.arithmeticOperator], color: "var(--syn-operator)" },
  { tag: [t.comment, t.lineComment, t.docString], color: "var(--syn-comment)", fontStyle: "italic" },
  { tag: [t.self, t.standard(t.variableName)], color: "var(--syn-builtin)" },
  { tag: [t.punctuation, t.paren, t.squareBracket, t.brace], color: "var(--syn-punct)" },
  { tag: t.propertyName, color: "var(--syn-name)" },
]);

const theme = EditorView.theme({
  "&": { backgroundColor: "var(--code-bg)", color: "var(--syn-name)", fontSize: "13.5px", borderRadius: "10px", border: "1px solid var(--line)" },
  "&.cm-focused": { outline: "none", borderColor: "var(--green)", boxShadow: "0 0 0 3px var(--green-soft)" },
  ".cm-scroller": { fontFamily: "var(--font-mono)", lineHeight: "1.62", minHeight: "200px", maxHeight: "560px" },
  ".cm-content": { padding: "12px 0", caretColor: "var(--ink)" },
  ".cm-gutters": { backgroundColor: "transparent", border: "none", color: "var(--gutter)" },
  ".cm-lineNumbers .cm-gutterElement": { padding: "0 14px 0 12px" },
  ".cm-activeLine": { backgroundColor: "color-mix(in srgb, var(--green-soft) 45%, transparent)" },
  ".cm-activeLineGutter": { backgroundColor: "transparent", color: "var(--green-ink)" },
  ".cm-selectionBackground, &.cm-focused .cm-selectionBackground, ::selection": { backgroundColor: "var(--selection) !important" },
  ".cm-cursor": { borderLeftColor: "var(--ink)", borderLeftWidth: "2px" },
  ".cm-matchingBracket": { backgroundColor: "var(--green-soft)", outline: "1px solid var(--green)" },
});

/** create(parent, doc, {onChange, onRun}) → EditorView */
function create(parent, doc, { onChange, onRun } = {}) {
  const runKeys = onRun ? [{ key: "Mod-Enter", run: () => (onRun(), true) }, { key: "Shift-Enter", run: () => (onRun(), true) }] : [];
  return new EditorView({
    parent,
    state: EditorState.create({
      doc,
      extensions: [
        lineNumbers(), highlightActiveLineGutter(), highlightSpecialChars(), history(), drawSelection(),
        indentOnInput(), bracketMatching(), closeBrackets(), highlightActiveLine(),
        indentUnit.of("    "), EditorState.tabSize.of(4),
        keymap.of([...runKeys, ...closeBracketsKeymap, ...defaultKeymap, ...historyKeymap, indentWithTab]),
        python(), syntaxHighlighting(highlight), theme,
        EditorView.updateListener.of((u) => { if (u.docChanged && onChange) onChange(u.state.doc.toString()); }),
        EditorView.contentAttributes.of({ "aria-label": "Code editor. Escape then Tab to leave." }),
      ],
    }),
  });
}

function setDoc(view, text) {
  view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: text } });
}

window.WBCodeMirror = { create, setDoc };
