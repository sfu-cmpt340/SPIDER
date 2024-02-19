
# Cell Detection from Cell-Tissue Interaction ![gc_banner_GzbItDZ x20](https://github.com/sfu-cmpt340/2024_1_project_02/assets/113268694/1890d570-5fa6-4a96-91d0-3700241e5e78)

OCELOT -- Overlapped Cell On Tissue Data Histopathology

# Introduction ✏️
Cell detection in histology images is one of the most important tasks in computational pathology. Recently, the OCELOT dataset is released in [dataset](https://openaccess.thecvf.com/content/CVPR2023/html/Ryu_OCELOT_Overlapped_Cell_on_Tissue_Dataset_for_Histopathology_CVPR_2023_paper.html) which provides overlapping cell and tissue annotations on images acquired from multiple organs stained with H&E. [dataset](https://openaccess.thecvf.com/content/CVPR2023/html/Ryu_OCELOT_Overlapped_Cell_on_Tissue_Dataset_for_Histopathology_CVPR_2023_paper.html) showed that understanding the relationship between the surrounding tissue structures and individual cells can boost cell detection performance.

With the newly released OCELOT dataset, our project is aim to promote research on how to utilize cell-tissue relationships for better cell detection. Unlike typical cell detection challenges, our project can utilize tissue patches and annotation for the purpose of boosting cell detection performance.

## Important Links 🔗

| [Dataset Download](https://zenodo.org/records/7844149) | [Dataset Website](https://lunit-io.github.io/research/ocelot_dataset/) | [Evaluation Metric](https://ocelot2023.grand-challenge.org/evaluation-metric/) | [Slack channel](https://app.slack.com/client/T06AP91EYG6/C06DW38TA3X) | [Project report](https://google.com) | [Timesheet](https://google.com)



- Dataset Download: Link to download the dataset of this project.
- Dataset Website: Link to the website of the dataset.
- Evaluation Metric: Link to evaluation we are using in this project.
- Timesheet: Link your timesheet (pinned in your project's Slack channel) where you track per student the time and tasks completed/participated for this project/
- Slack channel: Link your private Slack project channel.
- Project report: Link your Overleaf project report document.


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
