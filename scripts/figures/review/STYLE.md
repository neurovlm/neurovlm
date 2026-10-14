# Style guide for the NeuroVLM response letter

Sources: the author's hand edits to `response_letter.tex` (git diff), the paper (`sn-article.tex`), `check5.md` (General) and the author's standing rules. Apply it to every rewrite of the letter.

## Voice

- First person plural for what we did ("We retrained", "We added"). Passive is fine for standard method steps, as in the paper ("Coordinates were convolved with a 9 mm Gaussian kernel").
- Present tense for what a figure shows and what the model does ("Fig. R1 scores text-to-brain maps", "NeuroVLM leads"). Past tense for what was done.
- Simple, plain, direct. Answer exactly what the reviewer asked, in the reviewer's own terms. Go beyond the request only when the answer needs it.
- No hedges or intensifiers: "it is worth noting", "note that", "importantly", "notably", "interestingly", "arguably", "namely", "in other words", "indeed".
- Thank once, in the editor letter ("We thank the editor and reviewers for their thoughtful comments."). Do not open responses with praise of the comment.
- "We agree" only when we accept the point and changed something. The same or next sentence says what changed.
- Concede by stating the fact, then the fix: "The submission thresholded every map alike, after removing masks, parcellations, anatomical and variance maps." No apology and no self-criticism (the author cut "the paper did not say so").
- When we did not do something, say so in one sentence, give the reason in one clause (expert ratings are slow and hard to scale), and say what we did instead.
- No commentary about the response itself (cut: "where a request could not be met, the response says so", "we list them so that none passes unnoticed").

## Sentences

- Short. Aim for 10 to 20 words. The paper's median is 17 and the author's rewrites run 7 to 18. Split anything over 30.
- One idea per sentence. Cut the clause that restates a consequence ("so these cannot be separated"), the aside that cites precedent ("as in Neurosynth and NeuroQuery") and the bonus result nobody asked for ("it also leads against the raw maps").
- No semicolons. No colons as connectors. A colon may only end a line that introduces a list ("The main changes are:").
- No em dashes and no spaced en dashes. Use a period. Number ranges ("Boxes 1--5") are fine.
- At most one parenthetical per sentence, never nested, and only for a figure, a comment link or numbers: "(Fig. R2)", "(\cmt{1.3})", "(Dice 0.53 against at most 0.45)".
- No time markers for the revision ("Now p-maps are" became "P-maps are"). Present tense already says it is current.
- Name things in full and plainly: "a null baseline" not "a null", "Each map" not "The pooled map".
- Say what changed, not a vague direction: "reduces evaluation performance" replaced "scores lower".
- Define an acronym at first use, as the paper does: "dictionaries of functional modes (DiFuMo)".
- Serial comma, as in the paper ("titles, abstracts, and coordinate tables").
- American spelling. Seen in the paper: summarized, labeled, labeling, unlabeled, binarized, normalized, optimized, initialized, standardized, modeled, modeling, behavior, generalization, localization, organization, emphasize, minimize, nearest-neighbor. Also analyze, color, center, centered, favor, quantization, gray, while (not whilst), use (not utilise). The letter still has British forms to fix: summarised, labelled, labelling, unlabelled, centred, colour, binarised, binarises, favour, quantisation, nearest-neighbour.

## Paragraph shapes

- A response paragraph has at most 70 words and at most two numbers, in natural prose. Order:
  1. Answer. The first sentence answers the question directly ("No training stage uses a validation or test paper.").
  2. Evidence. The figure or table and at most two numbers ("(Fig. R2)", "Dice 0.53 against at most 0.45").
  3. Change. One sentence on what changed in the manuscript, naming the section.
- When the reviewer lists points (bullets, numbered items or several questions), answer each under a bold run-in label, in the reviewer's order, one paragraph per label. Sentence case, period inside the bold, numbered when there are several: `\textbf{1. Separation of datasets.}`, `\textbf{2. Positive and negative effects.}`. The author added `\textbf{Independent evaluation.}` in 1.1 as the first such label. Each labeled paragraph follows the same three-part shape and limits.
- A short opening paragraph after `\response` may frame the answer ("We agree. Fig. R1 scores ..."). The labeled paragraphs follow it.
- Letter to the editor: one sentence of thanks, one sentence of claim, then "The main changes are:" and an `itemize` list. One change per item, each a short sentence with a capital letter and a final period. Never an inline list joined by semicolons.
- No code paths, file names, function names or flags. Describe what the code does.

## Numbers

- At most two numbers per response paragraph. Comment and figure IDs (1.3, Fig. R2) do not count. Other values go in the figure or table.
- Two decimals for AUC, Dice, correlation and cosine (0.53, 0.62). Use three only to break a tie.
- Compare with "against" or "versus", never "vs": "AUC 0.62 against 0.50". Give the chance level when it is not obvious.
- Thousands commas for counts: 2,971 papers, 30,826 publications. Sample size as "n = 2,971" with spaces.
- Percent as `87\%` with no space (not `87~\%`, not "87 percent").
- Ranges with "to": "0.80 to 0.83".
- Spell out one to nine in prose, except with units, decimals or labels ("three seeds", "9 mm", "stage 1").

## Captions

- At most 45 words: an optional bold title of two to five words (the paper's form), then one sentence on what is plotted and one on how it is computed or read (axes, panels, bars, colors).
- No method details, sample breakdowns, hyperparameters, result narration or caveats. Those go to the Methods or the response text.
- Tables label rows and columns so they read without the caption ("Hard negatives", not "Same domain").
- The normalized-rank axis is labeled "k/n", and the caption defines it ("with k/n the fraction of all candidates"), as in Fig. 4 (check6). Figs. 5 and S3 still print the longer "k/n (fraction of all candidates)" until they are redrawn.
- Paper model (29 words): "\textbf{Gaussian products transformed to DiFuMo.} Points, from coordinate tables, were first convolved with a Gaussian kernel (top). The image output from Gaussian kernel convolution was reconstructed with DiFuMo (bottom)."
- Fig. R1 at this length (illustration, 32 words): "Text-to-brain maps from NeuroVLM and simpler methods, scored against each test paper's reconstructed map (n = 2,971). Panels show (a) Dice and (b) voxelwise Pearson r, with means and 95\% bootstrap intervals."

## Terms to use and avoid

| Use | Avoid |
|---|---|
| reviewers | the three reviewers |
| thoughtful comments | careful reading |
| The revisions and response address every comment. | The revision answers every comment; where a request could not be met, ... |
| a null baseline | a null |
| Each map | The pooled map |
| reduces evaluation performance (or name the metric) | scores lower |
| P-maps are inverted | Now p-maps are inverted |
| an `itemize` list after "The main changes are:" | an inline list joined by semicolons |
| final head, final model | submitted head, "(which beat the submitted head)" |
| (cut) | so these cannot be separated; as in Neurosynth and NeuroQuery; which are not maps of an effect; the paper did not say so; it also leads against the raw maps |
| (delete) | Some changes go beyond the reviewers' requests, and we list them so that none passes unnoticed. |
| test set | test PMIDs |
| map | spatial prediction |
| hard negatives | same domain, same-domain |
| linear (ridge) regression, with how it differs from the projection head | ridge regression alone, when the reviewer asked for linear regression |
| settings chosen on the validation set | choosing the recipe on the validation set only, recipe |
| calibrated (rescale only if calibration fails) | rescaled to peak at one, post hoc |
| 1.3: NeuroVault evaluation | Reviewer 1, comment 1.3: NeuroVault evaluation |
| k/n on the axis, defined in the caption as the fraction of all candidates | other labels for the normalized-rank axis |
| Qformer (as in the paper) | Q-Former |
| versus, against | vs |
| American forms listed under Sentences | summarised, labelled, labelling, unlabelled, centred, colour, binarised, favour, quantisation, whilst, utilise |

## Figure and comment references

- Response figures and tables: `Fig.~\ref{fig:label}` and `Table~\ref{tab:label}`, which render as "Fig. R3" and "Table R2". Never type the number and never write "Figure".
- Manuscript figures keep their own numbers: "Fig. 5b", "Fig. S3".
- Put the figure first as the subject ("Fig. R1 scores text-to-brain maps in voxel space") or in parentheses at the end of the sentence ("(Fig. R2)").
- Link other comments with `\cmt{1.3}`, written "(\cmt{1.3})" or "comment \cmt{1.3}".
- Headings: "1.3: NeuroVault evaluation". Number, colon, short noun phrase in sentence case, no question marks. Numbering is sequential (1.1, 1.2, ..., 2.1, ...), with no "1.m5", "1.code" or "3.assessment".
- Every figure and table in a response is cited in that response's text.

## Before and after, from the author's edits

| # | Before (replaced) | After (author) | What changed |
|---|---|---|---|
| 1 | We thank the editor and the three reviewers for their careful reading. | We thank the editor and reviewers for their thoughtful comments. | Count dropped. Thanks the comments, not the effort. |
| 2 | The revision answers every comment; where a request could not be met, the response says so and what we did instead. | The revisions and response address every comment. | Semicolon and meta clause cut. One claim in seven words. |
| 3 | The main changes are: independent voxel-space evaluation ...; a comparison of generated and retrieved captions; ...; a leakage audit beyond DOI matching; out-of-distribution and failure-case analyses; and corrected numbers, names and claims throughout. | The main changes are: [itemize] A comparison of generated and retrieved captions. / A leakage audit beyond DOI matching. / Out-of-distribution and failure-case analyses. / Corrected numbers, names and claims throughout. | Semicolon list became bullets. One change per item, capitalized, final period. |
| 4 | ... the mean map and a null, plus recall@k, MRR, median rank and hard negatives; | ... the mean map and a null baseline. Additional metrics report for recall@k, MRR, median rank and hard negatives. | Bare "a null" named in full. The "plus" add-on became its own sentence. Fix the slip as "We also report recall@k, ...". |
| 5 | NeuroVLM leads, narrowly ahead of ridge regression (which beat the submitted head) and ahead of mean and domain maps, ... | NeuroVLM leads, narrowly ahead of ridge regression and ahead of mean and domain maps, ... | Parenthetical about the old head cut. No submitted-versus-final contrast. |
| 6 | ... the nearest training paper's map (Dice 0.53 against at most 0.45); it also leads against the raw maps. | ... the nearest training paper's map (Dice 0.53 against at most 0.45). | Extra result after the semicolon cut. Two numbers left. |
| 7 | ... the released coordinates carry no experiment, contrast, table or sign labels, so these cannot be separated. | ... the released coordinates carry no experiment, contrast, table or sign labels. | Obvious consequence cut. The fact is enough. |
| 8 | The pooled map matches the paper-level text it is paired with, as in Neurosynth and NeuroQuery, and shows where effects lie, not their sign. | Each map matches the paper-level text it is paired with and shows where effects lie, not their sign. | Plain subject. Precedent aside cut. |
| 9 | The submission thresholded every map alike, after removing masks, parcellations, anatomical and variance maps, which are not maps of an effect; the paper did not say so. | The submission thresholded every map alike, after removing masks, parcellations, anatomical and variance maps. | Justification and self-criticism cut. Concede by stating the fact. |
| 10 | Now p-maps are inverted to 1 $-$ p and signed maps keep their positive tail, as before; using both tails, or adding negative tails as extra maps, scores lower. | P-maps are inverted to 1 $-$ p and signed maps keep their positive tail, as before; using both tails, or adding negative tails as extra maps, reduces evaluation performance. | "Now" dropped and the vague result made concrete. The semicolon the author kept still has to become a period. |

The author's edits also left `\end{itemize};` and a colon connector in 1.2 ("effect signs: the released coordinates"). Both break the rules above. Fix them in the next pass.
