# Papers for Team 38

The primary-source catalog is [manifest.json](manifest.json). PDFs are local research
materials and are excluded from Git; the catalog and downloader can be shared.

```bash
python3 scripts/download_papers.py
python3 scripts/download_papers.py --check-only
```

All five PDFs are now available locally. SimCLR was copied from the existing
course folder; the user ran the downloader for the other four after the session's
network restriction blocked direct retrieval. All five passed signature/checksum
checks and were opened successfully with PDFKit. Local page counts and SHA256
values are recorded in [downloads.json](downloads.json).

| Paper | Primary source | Read for this project |
|---|---|---|
| SimCLR, Chen et al., ICML 2020 | [Source](https://arxiv.org/abs/2002.05709), [local PDF](simclr-chen2020.pdf) | B.9 and Figure B.7; pretraining and linear evaluation |
| RotNet, Gidaris et al., ICLR 2018 | [Source](https://arxiv.org/abs/1803.07728), [local PDF](rotnet-gidaris2018.pdf) | CIFAR training and intermediate-layer evaluation in Section 3.1 |
| SimSiam, Chen & He, CVPR 2021 | [Source](https://arxiv.org/abs/2011.10566), [local PDF](simsiam-chenhe2021.pdf) | Stop-gradient, predictor, BN; CIFAR adaptation in Appendix D |
| ResNet, He et al., CVPR 2016 | [Source](https://arxiv.org/abs/1512.03385), [local PDF](resnet-he2016.pdf) | Shared CIFAR ResNet-20, Section 4.2 |
| LARS, You et al., 2017 | [Source](https://arxiv.org/abs/1708.03888), [local PDF](lars-you2017.pdf) | Large-batch optimizer implementation |

Read the local proposal first: it deliberately uses ResNet-20 for all three methods
and promises a reduced grid and trend reproduction. The original SimCLR figure
uses ResNet-50; its numerical accuracy is reference context, not a target guaranteed
for this project.

Additional implementation references:

- [Authors' SimCLR repository](https://github.com/google-research/simclr).
  Its README's single-GPU CIFAR example uses ResNet-18 and weight decay 1e-4;
  this is a different recipe from the paper's Figure B.7. Avoid silently combining them.
- [Authors' SimCLR learning-rate implementation](https://github.com/google-research/simclr/blob/master/tf2/model.py).
  Linear scaling multiplies the base rate by batch size / 256. Record the base,
  scaling rule, and actual peak rate separately.
- [Authors' SimSiam repository](https://github.com/facebookresearch/simsiam).
  Use as a reference for predictor/stop-gradient and frozen-evaluation behavior;
  its ImageNet scripts are not drop-in ResNet-20 CIFAR scripts.
- [DCC architecture](https://oit-rc.pages.oit.duke.edu/rcsupportdocs/dcc/architecture/),
  [scheduling](https://oit-rc.pages.oit.duke.edu/rcsupportdocs/dcc/slurm/), and
  [H200 access](https://oit-rc.pages.oit.duke.edu/rcsupportdocs/dcc/h200/).
  Verify current account access and live scheduler state rather than assuming a
  listed GPU is available to the team.
