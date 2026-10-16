# Explaining the project

## A one-minute answer

> In EEC 174, I worked with medical ventilator data to classify breathing patterns. I used flow and pressure signals to extract features such as breathing time, inhaled volume, and the ratio of exhaled to inhaled air. I compared Random Forest, Logistic Regression, and KNN, and implemented a rule baseline based on the definitions from the course. The Random Forest reached about 89% accuracy, but it detected only about 2% of the double-trigger cases. The rule baseline detected about 54%. That showed me why I need to look at each class, rather than choose a model based on overall accuracy. I later organized the work into a Python package with a runnable demo and patient-separated evaluation.

## Walk through it in this order

| Step | English explanation | 中文抓手 |
|---|---|---|
| Problem | “The patient's breathing and the ventilator's timing can be out of sync.” | 病人与机器的呼吸节奏不一致 |
| Input | “Each breath has a flow trace and a pressure trace.” | 一次呼吸对应两条信号 |
| Features | “I turn each trace into measurements with a physical meaning.” | 时长、体积、比例；不是随便凑数字 |
| Models | “I compare learned models with a simple rule baseline on the same validation set.” | 同一验证集才有可比性 |
| Finding | “Good overall accuracy can still miss most examples of a rare class.” | 89% 准确率掩盖 DTA 漏检 |
| Packaging | “I added a public demo and kept patients separate in the new evaluation.” | 新公开版有运行入口、患者分组和测试 |

## Useful follow-ups

**Why volume ratio?** “It compares the air breathed out with the air breathed in. It gives a feature with a clear physical meaning and lets me compare a rule with a learned model.”

**Why macro F1?** “It gives each class equal weight. Accuracy can be dominated by the most common class.”

**Why patient-separated validation?** “Breaths from the same patient can be similar. Keeping a patient in one split lets me test on different patients. The public package does that explicitly.”

**How do labels stay aligned?** “I join features and labels using both patient ID and breath ID. A shuffled file should not change which label belongs to a breath.”

**What was your contribution?** “The course supplied the data and notebook structure. I implemented the added features and experiments. The public package is my standalone implementation with clearer inputs, evaluation, tests, and results.”

**Which results are from the medical dataset?** “The table at the top of the README comes from the original saved experiment. The runnable demo generates its own waveforms, and its results are labelled separately.”
