# References

Course documents are kept locally in this folder and are not committed,
because the repository is public.

| Course document | Local file | Where to look |
|---|---|---|
| Team 38 proposal | `proposal.pdf` | Shared model, methods, and the planned comparisons |
| ECE 661 project guidelines | `course-guidelines.pdf` | Deliverables and presentation schedule |
| ECE 661 topic list | `course-topics.pdf` | Project 4 requirements, page 5 |

| Paper | What to read |
|---|---|
| [SimCLR · Chen et al., 2020](https://arxiv.org/abs/2002.05709) | Augmentation and training recipe, Figure B.7 batch/epoch comparison, Section 6 label-efficiency experiment |
| [RotNet · Gidaris et al., 2018](https://arxiv.org/abs/1803.07728) | Rotation task and CIFAR feature evaluation in Section 3.1, limited-label comparison in Figure 5 |
| [SimSiam · Chen and He, 2021](https://arxiv.org/abs/2011.10566) | Predictor and stop-gradient in Section 3, CIFAR recipe in Appendix D |
| [ResNet · He et al., 2016](https://arxiv.org/abs/1512.03385) | CIFAR ResNet-20 architecture in Section 4.2 |
| [LARS · You et al., 2017](https://arxiv.org/abs/1708.03888) | Layer-wise learning-rate scaling for large batches |

Run `python scripts/download_papers.py` from the repository root to fetch
the Git-ignored PDFs. Add `--check-only` to inspect local copies. The
[catalog](papers/manifest.json) lists filenames and source URLs. The
[download record](papers/downloads.json) contains the recorded checksums.

Implementation references are available in the
[SimCLR repository](https://github.com/google-research/simclr), its
[learning-rate scheduler](https://github.com/google-research/simclr/blob/master/tf2/model.py),
and the [SimSiam repository](https://github.com/facebookresearch/simsiam).
