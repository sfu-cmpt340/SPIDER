
# Spine Segmentation![SPIDER_banner_F4wHbpf x20](https://github.com/sfu-cmpt340/2024_1_project_02/assets/113268694/c3f9ff84-e48e-400e-81f6-0014770a028e)

# Introduction ✏️
Low back pain (LBP) is a significant cause of disability and a major societal healthcare issue. One of the commonly used diagnostic and treatment decision-making tools for LBP is magnetic resonance imaging (MRI) of the lumbar spine. 
The project focuses on the segmentation of three anatomical structures in lumbar spine MRI: vertebrae, intervertebral discs (IVDs), and spinal canal. The segmentation task aims to produce separate masks for each vertebra, IVD, and the spinal canal in the lumbar spine MRI volume. The numbering of the vertebrae and IVDs is not specific and may vary across different cases.

## Important Links 🔗

| [Dataset Download](https://zenodo.org/records/8009680) | [Evaluation Metric](https://spider.grand-challenge.org/segmentation-evaluation/) | [Slack channel](https://app.slack.com/client/T06AP91EYG6/C06DW38TA3X) | [Project report](https://google.com) | [Timesheet](https://google.com) |



- Dataset Download: Link to download the dataset of this project.
- Evaluation Metric: Link to evaluation we are using in this project.
- Slack channel: Link to private Slack project channel.
- Project report: Link to Overleaf project report document.
- Timesheet: Link to timesheet (pinned in your project's Slack channel) where you track per student the time and tasks completed/participated for this project/


## Video/demo/GIF
Record a short video (1:40 - 2 minutes maximum) or gif or a simple screen recording or even using PowerPoint with audio or with text, showcasing your work.


## Table of Contents
1. [Demo](#demo)

2. [Installation](#installation)

3. [Reproducing this project](#repro)

4. [Guidance](#guide)


<a name="demo"></a>
## 1. Example demo

A minimal example to showcase your work

```python
from amazing import amazingexample
imgs = amazingexample.demo()
for img in imgs:
    view(img)
```

### What to find where

Explain briefly what files are found where

```bash
repository
├── src                          ## source code of the package itself
├── scripts                      ## scripts, if needed
├── docs                         ## If needed, documentation   
├── README.md                    ## You are here
├── requirements.yml             ## If you use conda
```

<a name="installation"></a>

## 2. Installation

Provide sufficient instructions to reproduce and install your project. 
Provide _exact_ versions, test on CSIL or reference workstations.

```bash
git clone $THISREPO
cd $THISREPO
conda env create -f requirements.yml
conda activate amazing
```

<a name="repro"></a>
## 3. Reproduction
Demonstrate how your work can be reproduced, e.g. the results in your report.
```bash
mkdir tmp && cd tmp
wget https://yourstorageisourbusiness.com/dataset.zip
unzip dataset.zip
conda activate amazing
python evaluate.py --epochs=10 --data=/in/put/dir
```
Data can be found at ...
Output will be saved in ...

<a name="guide"></a>
## 4. Guidance

- Use [git](https://git-scm.com/book/en/v2)
    - Do NOT use history re-editing (rebase)
    - Commit messages should be informative:
        - No: 'this should fix it', 'bump' commit messages
        - Yes: 'Resolve invalid API call in updating X'
    - Do NOT include IDE folders (.idea), or hidden files. Update your .gitignore where needed.
    - Do NOT use the repository to upload data
- Use [VSCode](https://code.visualstudio.com/) or a similarly powerful IDE
- Use [Copilot for free](https://dev.to/twizelissa/how-to-enable-github-copilot-for-free-as-student-4kal)
- Sign up for [GitHub Education](https://education.github.com/) 
