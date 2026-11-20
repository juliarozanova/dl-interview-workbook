---
id: ch26-ablations
summary: "Research rounds test judgement more than code. Design the ablations that would convince a sceptical reviewer, choose metrics for a problem where accuracy is meaningless, and see how an unequal tuning budget manufactures a fake improvement."
udl: "—"
---

## Convince a sceptic

A research interviewer's favourite question has the form "*Your method beats the baseline by X. Why should I believe it, and what is actually doing the work?*" Strong answers share a structure:

1. **Rule out the boring explanations first.** Different compute, more tuning, a better baseline implementation, data leakage, a lucky seed.
2. **Isolate components.** Remove (or swap) one piece at a time, with everything else held fixed.
3. **Quantify uncertainty.** Seeds, confidence intervals, and more than one dataset (Chapter 24).
4. **Test the mechanism.** If you claim *why* it works, design an experiment that would come out differently if you were wrong.

The two **design** exercises here have no single right answer. Talk through yours, then tick off the rubric honestly. The predict exercise shows numerically how unequal tuning manufactures gains.

::::exercise{id=ablation-design type=design difficulty=2 time=10 runtime=none udl="—" tags="ablations,experimental-design"}
# Design the ablations

:::task
Your new method for machine translation adds **two things** to a strong transformer baseline: (a) a new auxiliary loss, and (b) a data-augmentation scheme. It improves BLEU by **+1.2** on one benchmark. You trained 3 seeds of your method; the baseline numbers come from the original paper. An interviewer asks: **"What experiments would you run before believing this?"**
:::

:::rubric
- **Reproduce the baseline yourself**, with the same codebase, data, compute and tuning budget, rather than copying the paper's number.
- **Ablate each component separately:** baseline + (a) only, baseline + (b) only, and both. Check whether they're additive or interacting.
- **Match compute and tuning:** the same number of training steps or FLOPs, and the same hyperparameter-search budget for every arm (see Exercise 26.3).
- **Report variance:** multiple seeds per arm, confidence intervals or a paired test, and the seed standard deviation next to the +1.2.
- **Go beyond one benchmark and one metric:** other language pairs or test sets, plus a second metric (chrF, COMET, human evaluation), since BLEU can be gamed.
- **Probe the mechanism:** for example, does the auxiliary loss help more on long sentences or rare words, as your story predicts? Pick an analysis whose result *could* contradict the claimed explanation.
- **Check for leakage or contamination:** does the augmentation draw on data that overlaps the test set?
:::

:::answer
**Start with the baseline.** A +1.2 against a number copied from a paper confounds the method with everything else that differs: code, preprocessing, hardware, tuning effort. Re-run the baseline in *your* pipeline, with the *same* budget. Often half of the gain disappears here.

**Then a 2×2 ablation.** Baseline; +loss; +augmentation; +both. This tells you which component matters, and whether they interact. If "+both" ≈ "+augmentation", the new loss (probably your headline contribution) isn't doing the work. Keep training steps or FLOPs fixed across all four arms. If the augmentation makes each epoch bigger, decide explicitly whether you're matching epochs or compute, and say which.

**Uncertainty.** With 3 seeds, give means ± standard deviation and a paired bootstrap interval over test sentences (Chapter 24). If the seed standard deviation is 0.6 BLEU, then +1.2 from three seeds is suggestive, not conclusive.

**Breadth.** One benchmark is one draw. Add a second language pair and a learned metric such as COMET. If possible, run a small human evaluation, because BLEU improvements below about 1 point often don't survive human judgement.

**Mechanism.** State the hypothesis ("the loss improves the handling of rare words"), then test a prediction that could fail: bucket the gains by word frequency, or remove the rare words from the auxiliary loss's targets and check that the gain shrinks.

**Contamination.** If the augmentation uses external monolingual data or back-translation, check it for overlap with the test set.

*How to say it in the room:* "First I'd make sure the comparison is fair, then find out which part matters, then check the effect is bigger than the noise, then test the story behind it."
:::

:::interview
- What's the difference between an ablation and a sensitivity analysis?
- Your two components only help together. How do you report that?
- How many seeds is enough? *(Enough that the confidence interval excludes zero by a margin you'd bet on. Estimate the seed variance first.)*
:::
::::

::::exercise{id=metric-design type=design difficulty=2 time=8 runtime=none udl="—" tags="metrics,experimental-design"}
# Choose the metrics: rare-event detection

:::task
You're building a classifier that flags **fraudulent transactions**. About **0.2%** of transactions are fraud. A human team can review **500 flagged transactions a day**, out of 1 million. Your colleague reports **99.8% accuracy**. **Which metrics would you use instead, and how would you choose the operating threshold?**
:::

:::rubric
- **Explain why accuracy is meaningless here:** predicting "not fraud" for everything scores 99.8%.
- **Use precision and recall,** or the precision–recall curve and **AUPRC**, rather than AUROC alone, which can look great under extreme imbalance.
- **Tie the threshold to the review budget:** precision and recall at the top 500 per day (precision@k / recall@k).
- **Bring in costs:** the cost of a missed fraud versus the cost of a wasted review, and choose the threshold that minimises expected cost.
- **Check calibration** if the scores feed a decision rule, or are shown to the reviewers (Chapter 24's ECE).
- **Evaluate over time:** a temporal split (train on the past, test on the future), plus monitoring for drift. Fraud patterns adapt.
:::

:::answer
**Accuracy is the wrong question.** At a 0.2% base rate, the constant "never fraud" model scores 99.8%. Your colleague's number says nothing.

**Precision and recall, at the operating point that matters.** The team reviews 500 cases a day out of about 1M transactions, around 2,000 of which are fraud. So the natural metrics are **precision@500** (the fraction of reviews that are real fraud) and **recall@500** (the fraction of all fraud caught). Over the whole range, report the **precision–recall curve** and AUPRC. AUROC is dominated by the huge number of easy negatives, and can sit at 0.98 while precision at any usable threshold is poor.

**Costs set the threshold.** If a missed fraud costs about $500 and a review about $5, choose the threshold that minimises expected cost, subject to the 500-per-day limit. Make that trade-off explicit to stakeholders.

**Calibration** matters if the scores are shown to reviewers or combined with other signals. Temperature-scale on recent data.

**Time-based evaluation.** Fraud is adversarial and drifts. Train on months 1–6, evaluate on month 7, and monitor precision@k in production with delayed labels. A random split would leak future patterns into training (Chapter 25).

*In one sentence:* "I'd optimise and report precision at the review capacity, choose the threshold from costs, and evaluate on future data, because accuracy, and even AUROC, hide what matters here."
:::

:::interview
- Precision versus recall: give a problem where you'd favour each.
- Why can AUROC mislead under heavy imbalance?
- How would you evaluate a model whose labels arrive with a 30-day delay?
:::
::::

::::exercise{id=tuning-budget-predict type=predict difficulty=2 time=6 runtime=browser udl="—" tags="ablations,significance,experimental-design"}
# The unfair tuning budget

:::task
Your method and the baseline are **actually equally good**. Each training run's validation score is the true score plus noise with a standard deviation of **0.5 points**. You tune your method with **50** hyperparameter trials, and the baseline with **5**, reporting the best trial for each. On average, how much "better" does your method look?
:::

:::choices
- [ ] 0 points: they're equally good
- [x] About 0.5 points
- [ ] About 2.5 points
- [ ] Your method looks *worse*: more trials means more bad configurations
:::

:::experiment{lang=python}
import numpy as np

rng = np.random.default_rng(0)

def best_of(k, sigma=0.5, reps=20000):
    return (sigma * rng.normal(size=(reps, k))).max(axis=1)

for k in (1, 5, 20, 50):
    print(f"best of {k:2d} trials: mean apparent score {best_of(k).mean():+.3f}")
gap = best_of(50).mean() - best_of(5).mean()
print(f"\napparent gain of 50-trial method over 5-trial baseline: {gap:+.3f} points (true gain 0)")
:::

:::tests{lang=python}
def test_gap_is_about_half_a_point():
    gap = best_of(50).mean() - best_of(5).mean()
    assert 0.45 < gap < 0.65

def test_more_trials_always_look_better():
    means = [best_of(k).mean() for k in (1, 5, 20, 50)]
    assert all(a < b for a, b in zip(means, means[1:]))
:::

:::explain
Reporting the **best** of $k$ noisy runs is reporting the maximum of $k$ draws. For Gaussian noise, $\mathbb E[\max_k] \approx \sigma\sqrt{2\ln k}$ for large $k$. Exactly, it's about $1.16\sigma$ for $k = 5$ and $2.25\sigma$ for $k = 50$. With $\sigma = 0.5$, that's a phantom gain of about **0.55 points**, created entirely by the tuning budget.

That's comparable to many published improvements. It's why careful papers:

- give every method the **same tuning budget** (or report performance as a function of budget);
- select on validation and **report on a separate test set**, so the maximum's optimism doesn't carry over;
- re-run the chosen configuration with **fresh seeds**, to measure its real mean.

**In the interview**, "Was the baseline tuned as hard as the proposed method?" is the single highest-value question to ask about any results table.
:::

:::interview
- Why is "best of $k$ runs" biased? How does the bias grow with $k$? *(Roughly $\sigma\sqrt{2\ln k}$.)*
- How would you compare methods fairly when one has more hyperparameters?
- What's the difference between validation-set overfitting and test-set overfitting, and how do benchmarks suffer from both?
:::
::::
