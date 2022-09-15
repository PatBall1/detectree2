.. _usinggit:

==========
Git/Github
==========

Overview
========

The following section provides a walkthrough of using git/GitHub.

Fork the repository:

Only the primary project team has permissions to directly edit the main detectree2 repository. As an external contributor, you will need to make a personal copy of the repository (a.k.a. “fork”) to begin making changes. To create a fork, click on the Fork button at the top right of the page of the repository on Github.

This will create a copy of the detectree2 repository under your own GitHub account. (Visit “Your Repositories” under your account.)

Clone the repository: 
To access and edit your fork (copy) of the repository on your computer, you will need to clone the repository.::

    git clone https://github.com/<YOUR-USERNAME>/detectree2.git detectree2-<YOUR-USERNAME>
    cd detectree2-<YOUR-USERNAME>
    git remote add upstream https://github.com/patball1/detectree2

Checkout a new branch:

While this step is optional for forked repositories, it's generally best practice to create a new branch for each thing you're working on. Run the following command::

    get checkout -b <NEW-BRANCH>

where <NEW-BRANCH> is the name of the branch. (For future contributions, replace this with a name of your choosing). This command creates the new branch and checks out the branch.

Committing your changes: 

To save this changes to Git, you must commit the files. Run the command::

    git commit -m "Add to README title"

where the text inside the quotes is a description of the changes you've made in this commit.

Pushing your changes: 

To push your changes online for the first time, run::

    git push origin --set-upstream update-readme

where --set-upstream update-readme creates the same update-readme branch online, in your fork on GitHub.

Creating a Pull Request: To submit your changes for review, you will need to create a pull request. When a new branch is pushed to GitHub, a prompt will appear to create a pull request:

Then fill out the form with (at minimum) a title and description and create the pull request. At this point, a project team member or maintainer can review your pull request, provide comments, and officially merge it when approved.


More advanced Git Practices
===========================

Rebase
------
If you're working on some changes for a long period of time, it's possible that other contributors may have submitted other changes on the same files you're working on (see section General Tips). To sync your branch, run::

    git pull origin master --rebase

and follow the prompts to approve and/or resolve the changes that should be kept.  Rebasing often proceeds as follows (taken from gitlab's docs). Fetch the latest changes from main::

    git fetch origin master

Checkout your feature branch::

    git checkout my-feature-branch

Rebase it against master::

    git rebase origin/master

Then force push to your branch.::

    git push -f origin my-feature-branch

When you rebase:

1. Git imports all the commits submitted to main after the moment you created your feature branch until the present moment.

2. Git puts the commits you have in your feature branch on top of all the commits imported from main:

It's worth bearing in mind that this can lead to issues especially when working with other people. There is a high chance of overwriting commits from your colleagues resulting in lost work. The safer alternative is to use --force-with-lease.::

    git push --force-with-lease origin my-feature-branch

Updating your Fork
------------------
To update the master branch of your fork (so that new branches are created off of an up-to-date master branch), run::

    git fetch upstream

Pull Request Conventions
------------------------
The pull request title is used to build the release notes. Write the title in past tense, describing the extent of the changes.

Pull Requests should have labels to identify which category belongs to in the release notes. Use the exclude notes label if the change doesn't make sense to document in release notes.

Pull Requests should be linked to issues, either manually or using keywords.
