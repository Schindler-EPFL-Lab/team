# TEAM: teach a robot arm to move

[![Build Status](https://dev.azure.com/devsdb/CRD-NT_ARCO/_apis/build/status/SchindlerReGIS.team?repoName=SchindlerReGIS%2Fteam&branchName=main)](https://dev.azure.com/devsdb/CRD-NT_ARCO/_build/latest?definitionId=1211&repoName=SchindlerReGIS%2Fteam&branchName=main)

![Schindler logo](logo.jg)

![ABB robot with a helmet](robot.jpg)
![ABB robot with a helmet](flowchart.png)

TEAM is a parameter-free algorithm to learn motions from user demonstrations.
[The paper](https://arxiv.org/abs/2209.06940) was published at the International Conference on Informatics in Control, Automation and Robotics.

If you use this work please cite:

```bib
@inproceedings{Panchetti2022TEAMAP,
  title={TEAM: a parameter-free algorithm to teach collaborative robots motions from user demonstrations},
  author={Lorenzo Panchetti and Jianhao Zheng and Mohamed Bouri and Malcolm Mielle},
  booktitle={International Conference on Informatics in Control, Automation and Robotics},
  year={2023},
  url={https://api.semanticscholar.org/CorpusID:252280502}
}
```

## License

Our objective is to share our results with the research community and produce reproducible science.
We wish that those who benefit from our work do the same :)

This package is licensed under the [Prosperity Public License 3.0 and the Parity Public License 7.0.0](LICENSE.md).

That means that this package is free to use for __non-commercial projects__---personal projects, public benefit projects, research, education, etc.
If your project is commercial (even for internal use at your company), you have 30 days to try this package for free before you have to contact Schindler for a licensing fee.
Furthermore, if you work on a non-commercial project, the Parity license means that this project is under a [maximal copyleft license](https://blueoakcouncil.org/copyleft).
Taken from the license file:

> Contribute software you develop, operate, or analyze with this software, including changes or additions to this software. When in doubt, contribute.

See the [License](LICENSE.md) file for all details.

## Build and Test

1. Intall [rws2](https://github.com/SchindlerReGIS/rws2).
   It's the dependency needed to control an ABB robot for testing purpose.
2. Install [flit](https://github.com/pypa/flit) with `pip install flit`.
We use flit to package and install this repository.
3. Clone/fork the repo from Github.
4. Run `pip install -e .` in the root folder to install rws2 in editable mode (`pip install .` is enough if you do not plan to contribute).

The library should then be installed and you should be able to call it in python with `import team`.

## How to use the package

See `notebook/pipeline.ipynb`.

## Contribute

PR request on GitHub are welcome.
We use [black](https://github.com/psf/black) for code formatting and [flake8](https://github.com/pycqa/flake8) for linting.
Code that do not follow black formatting and follow flake8 linting will be rejected by the pipeline.

A standard git commit message consists of three parts, in order: a summary line, an optional bod.
The parts are separated by a single empty line.
The summary line is included in the short logs (git log --oneline, gitweb, Azure DevOps, email subject) and therefore should provide a short yet accurate description of the change.
The summary line is a short description of the most important changes. The summary line must not exceed 50 characters, and must not be wrapped. The summary should be in the imperative tense.
The body lines must not exceed 72 characters and can describe in more details what the commit does.
