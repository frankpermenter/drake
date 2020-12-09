# -*- python -*-
def conex_repository(
        name,
        mirrors = None):
    native.local_repository(
        name = name,
        path = "/home/frank/conexnew/conex",
    )
