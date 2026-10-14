# Report finding codes

Stable enum of `findings[].code` values used in JSON reports. Codes are never renamed or reused; deprecated codes stay listed.

| Code | Tool | Severity | Meaning |
|---|---|---|---|
| `EVAL_CLASS_ABSENT` | eval | warn | A class has no pixels in the labels or the predictions, so it's excluded from mIoU. Expected only on small subsets. |
