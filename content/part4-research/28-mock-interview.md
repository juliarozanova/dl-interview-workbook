---
id: ch28-mock-interview
summary: "The spoken rounds: critiquing a paper, walking through your own research, and proposing new work. Each comes with a time box, a structure, and a rubric of what strong candidates cover. Do them out loud, ideally to another person."
udl: "—"
---

## Putting it together

The last weeks of the plan are about **performance under interview conditions**. The material is in the earlier chapters. These exercises train the *delivery*:

| Round | What they're testing | Practice |
|---|---|---|
| ML theory, rapid-fire | depth and recall under time pressure | the **drill deck**, plus **interview mode** on any chapter |
| Coding without frameworks | fluency and edge cases | Chapter 27 with interview mode on |
| Paper critique | research taste, scepticism, fairness | Exercise 28.1 |
| Your own research | clarity, ownership, judgement | Exercise 28.2 |
| Research proposal / design | scoping, experimental design | Exercise 28.3 and Chapter 26 |

**How to use the rubrics.** Answer out loud against the clock *first*, without peeking. Then tick only what you actually said. Repeat the same exercise a week later and compare. Recording yourself is uncomfortable, and very effective.

::::exercise{id=paper-critique type=design difficulty=2 time=15 runtime=none udl="—" tags="experimental-design,ablations"}
# Critique a paper in 15 minutes

:::task
Pick a recent paper from your target team, or any paper you know well. In **15 minutes**, as you would in an interview, give: (1) a two-sentence summary of the claim, (2) its strongest contribution, (3) its weakest point, and (4) the **one experiment** you'd ask the authors to add. Be fair: interviewers want critical *and* generous.
:::

:::rubric
- **State the claim precisely,** in your own words and in two sentences: what's new, and on which evidence it rests.
- **Name the real contribution,** not the abstract's framing: is it the idea, the engineering, the data, or the evaluation?
- **Check the baselines:** are they strong, tuned with the same budget, and run in the same codebase? Is the compute comparable?
- **Check the evidence:** seeds or error bars, the number of datasets, ablations that isolate the claimed mechanism, and any sign of test-set tuning.
- **Probe generality:** would it hold at a different scale, on other domains or data regimes? What's the most likely failure mode?
- **Propose one decisive experiment** that could *falsify* the main claim, and say what result you'd expect.
- **Stay balanced:** acknowledge what's genuinely good before the critique, and separate fatal flaws from nitpicks.
:::

:::answer
**A structure that works in 15 minutes:**

1. **Claim (1 minute).** "They claim X improves Y on Z, *because* of mechanism M." Including the "because" shows you read past the abstract.
2. **What's genuinely good (2 minutes).** Be specific: "The ablation in Table 3 cleanly isolates M." Interviewers notice candidates who can't say anything positive.
3. **Evidence audit (5 minutes).** Go through the usual suspects. Baseline strength and tuning parity. Compute parity. Variance (are there seeds or confidence intervals?). Benchmark breadth. Leakage or contamination (especially for language-model benchmarks). Whether the ablations actually isolate M or just remove things.
4. **Biggest weakness (3 minutes).** Choose *one* and argue it well, rather than listing ten. Classify it: does it threaten the claim, or only its scope?
5. **The decisive experiment (3 minutes).** Something whose outcome would change your mind. "If M is the cause, then removing M while matching compute should eliminate the gain on the long-context split. I'd expect a drop of at least half; if there's no drop, the gain comes from something else."
6. **Summary (1 minute).** "Strong idea, under-supported by the evidence for the mechanism. I'd accept it with the extra ablation."

**Common failure modes:** summarising for 10 minutes; critiquing only the writing; demanding experiments that are infeasible; being purely negative.
:::

:::interview
- What's the most common weakness in ML papers you read?
- How do you tell an engineering contribution from a scientific one?
- What makes an experiment *decisive*?
:::
::::

::::exercise{id=research-walkthrough type=design difficulty=2 time=10 runtime=none udl="—" tags="experimental-design"}
# Walk through your own research

:::task
"Tell me about the research you're most proud of." Give a **5-minute** walkthrough to an expert who isn't in your sub-field, then handle the follow-ups: "What would you do differently?", "What was *your* contribution?", and "What's the biggest limitation?"
:::

:::rubric
- **Lead with the problem and why it matters,** in one or two sentences, before any method details.
- **Give the key insight in a single sentence:** the idea that made it work, which a smart non-specialist can follow.
- **Make your role explicit:** what *you* did, as distinct from collaborators, with "I" for your parts.
- **Back it with evidence,** the one or two results that matter, with their baselines and uncertainty, rather than every table.
- **Volunteer a real limitation** and what it implies, before you're asked.
- **Say what you'd do differently or next,** showing that you've reflected beyond the paper.
- **Fit it in about 5 minutes,** with a natural pause for questions, and pitch it at the listener's level (check in rather than lecture).
:::

:::answer
**Use the five-sentence skeleton, then expand each sentence as time allows:**

1. **Problem:** "Models do X badly when Y, which matters because Z."
2. **Insight:** "We realised that ___, so instead of ___ we ___." *This is the sentence they'll remember. Practise it until it's crisp.*
3. **What I did:** "I designed ___ and ran ___. My co-author built ___."
4. **Evidence:** "On ___ it improves ___ by ___ over a tuned ___, across N seeds. And the ablation shows the gain comes from ___."
5. **Limitation and next step:** "It doesn't yet handle ___, because ___. Next I'd try ___."

**Prepare the follow-ups in advance.** The hardest question about your weakest result. The baseline you didn't run. What you'd do with 10× the compute. What you got wrong along the way. Interviewers probe for *ownership* and *judgement*: whether you understand why things worked, not just that they did.

**Delivery.** Draw one diagram. Avoid acronyms until you've defined them. Stop at 5 minutes and ask "Shall I go deeper on the method or the results?"
:::

:::interview
- What was the hardest technical problem in the project, and how did you get past it?
- If you started again today, what would you change?
- Which result are you least confident in?
:::
::::

::::exercise{id=research-proposal type=design difficulty=3 time=15 runtime=none udl="—" tags="experimental-design,ablations"}
# Propose a research project

:::task
"If you joined us, what would you work on in your first six months?" Propose a concrete project relevant to the team. In **10–15 minutes**, cover: the question, why now, the approach, how you'd know it worked, and what could go wrong.
:::

:::rubric
- **Pose a sharp, falsifiable research question,** not a topic ("Does X improve Y under Z?" rather than "work on X").
- **Say why it matters and why now:** a gap in current methods, or new capabilities, data or scale that make it feasible.
- **Give a concrete first experiment** that can be run in weeks, with a clear baseline and metric (start small).
- **Define success criteria in advance:** what result would count as success, failure or "interesting".
- **Identify risks and fallbacks:** the most likely way it fails, and the useful result you'd still get.
- **Be realistic about resources:** compute, data and collaborators, scoped to six months with milestones.
- **Connect it to the team's work:** how it builds on or complements what they already do.
:::

:::answer
**A template** (aim for 1–2 minutes per part):

1. **Question.** "Can ___ reduce ___ for ___, without hurting ___?" One sentence, falsifiable.
2. **Why now.** "Current approaches assume ___, which breaks when ___. Recent work on ___ makes a cheap test possible."
3. **Approach.** The simplest version first: "Month 1: reproduce ___ and establish the baseline on ___. Month 2: test ___ at small scale."
4. **Evaluation.** "Primary metric ___ on ___; a held-out environment ___ to rule out overfitting; three seeds; a compute-matched baseline." Borrow from Chapters 24–26.
5. **Decision points.** "If the small-scale effect is below ___, I'd stop and write up the negative result. If it's above ___, scale to ___."
6. **Risks.** "The most likely failure is ___. Even then, the analysis of ___ would be useful for ___."
7. **Fit.** "This would plug into your work on ___, and I'd want to collaborate with ___ on ___."

**What interviewers look for:** scoping (not boiling the ocean), a plan that produces signal early, honest risk assessment, and evidence that you've thought about *their* problems.
:::

:::interview
- What's the smallest experiment that would tell you whether this is worth pursuing?
- What would make you abandon the project?
- How does this relate to [a recent paper from the team]?
:::
::::
