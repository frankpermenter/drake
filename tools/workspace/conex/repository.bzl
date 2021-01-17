# -*- python -*-
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
#def conex_repository(
#        name,
#        mirrors = None):
#    native.local_repository(
#        name = name,
#        path = "",
#    )
def conex_repository(
        name,
        mirrors = None):
      git_repository(
          name = name,
          commit = "644fa61a02d930fb30880889bb7428d2ca94fb4b",
          shallow_since = "1610833788 -0500",
          remote = "git@github.com:frankpermenter/conex.git",
      )
