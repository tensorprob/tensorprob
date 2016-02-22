
# Contributing to TensorProb

TensorProb is an open source project, and we welcome contributions of all kinds!
For example, you can contribute by creating [issues][issues] or by submitting
pull requests.
See below for instructions on both of these.

## Submitting bug reports or suggestions

If you find a **bug** in tensorprob, it would be great if you could open an issue
on our [issue tracker][issues]. Before submitting the report, you should make
sure that the problem still persists in the newest version of `tensorprob` from
the `master` branch by running
```bash
git clone https://github.com/ibab/tensorprob
```
and installing the python package in the `tensorprob` directory, for example with
```bash
pip install --user --upgrade ./tensorprob
```
When filing the issue, please try to state exactly how to reproduce the problem.

If you have an **idea** for how to improve tensorprob, but you don't have enough
time or expertise to implement it yourself, we would still like to hear about it
in an [issue][issues]!

## Submitting contributions

By contributing, you are agreeing that we may redistribute your work under
[this license][license].

We use the [fork and pull][gh-fork-pull] model to manage changes. More information
about [forking a repository][gh-fork] and [making a pull request][gh-pull].

To contribute to this repository:

1. Fork [the project repository](https://github.com/ibab/tensorprob/).
   Click on the 'Fork' button near the top of the page. This creates a copy of
   the code under your account on the GitHub server.
2. Clone this copy of the repository to your local disk:

        $ git clone git@github.com:YourLogin/tensorprob.git
        $ cd tensorprob

2. Create a new branch in your clone `git checkout -b my-new-branch`. Never
   work in the ``master`` branch!
4. Work on this copy on your computer and use `git` to track your changes:

          $ git add modified_files
          $ git commit

   to record your changes in `git`. Please try to use informative commit
   messages!  Then, you can push your changes to GitHub with:

          $ git push -u origin my-new-branch

Before opening a pull request, please make sure that you've followed the
following **checklist**:

 - [ ] If you have added a new feature, please make sure that you have provided
   tests for that feature. We aim at achieving 100% test coverage for the project.
 - [ ] If you have fixed a bug, add a test to make sure that the bug doesn't
   happen again.

If you've followed these points, or you need assistance with the implementation,
you may proceed to open a pull request by:

 - Going to the web page of your fork of the `tensorprob` repo,
 - Clicking 'Pull request' to send your changes to the maintainers for
review. This will send an email to the maintainers.

After creating the pull request, it would be great if you could respond to
suggestions or other comments we have to your changes.
This will improve the chances of your pull request being accepted into the
project.

[issues]: https://github.com/ibab/tensorprob/issues
[license]: LICENSE
[gh-fork]: https://help.github.com/articles/fork-a-repo/
[gh-pull]: https://help.github.com/articles/using-pull-requests/
[gh-fork-pull]: https://help.github.com/articles/using-pull-requests/#fork--pull
