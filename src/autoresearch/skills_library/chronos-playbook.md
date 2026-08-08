---
name: chronos-playbook
description: Use Chronos responsibly for zero-shot forecasting, context tuning, quantiles, or lightweight fine-tuning.
---
Establish zero-shot performance before attempting fine-tuning.

- Match context length to the longest useful seasonal cycle while staying within memory and runtime limits.
- Batch series and use the smallest checkpoint likely to meet the budget.
- Use the median quantile for symmetric point metrics; choose another quantile only when the objective explicitly values asymmetric risk.
- Normalize or scale each series consistently and restore units before scoring.
- Fine-tune only when there are enough related series and repeated windows to avoid memorizing one validation period.
- Fall back to seasonal naive for too-short histories and consider blending Chronos with a tree or statistical model when errors are complementary.
