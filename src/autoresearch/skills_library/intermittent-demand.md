---
name: intermittent-demand
description: Handle sparse series with many zero-demand periods and irregular nonzero demand.
---
First measure zero share, nonzero interval length, and demand-size variability by SKU.

- Use Croston or SBA for stable intermittent demand and TSB when demand occurrence probability changes over time.
- Consider a hurdle model: forecast probability of nonzero demand separately from positive demand size.
- Pool information across related SKUs when individual series are too short.
- Avoid ordinary MAPE on zero actuals; use the configured protected metric and inspect WMAPE plus bias.
- Segment smooth, erratic, intermittent, and lumpy SKUs, then choose a fallback per segment.
- Do not round forecasts unless the business contract requires integer units; rounding can add systematic bias.
