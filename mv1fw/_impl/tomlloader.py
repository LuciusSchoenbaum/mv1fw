


from os.path import (
    join as os_path_join,
    exists as os_path_exists,
)
import tomli



class TomlLoader:
    """
    `TOML <https://toml.io/en/>`_ reader that wraps
    `tomli <https://pypi.org/project/tomli/>`_.

    """


    def __init__(self):
        pass


    def load(self, path_dir, path_fname, verbose = False):
        path = os_path_join(path_dir, path_fname)
        self.msg(f"load path {path}", verbose)
        read_ok, parse_ok, msg, tgtdict = self._read_parse_procedure(path, verbose)
        if not (read_ok & parse_ok):
            eemsg = f"load file: {msg}."
            raise EnvironmentError(eemsg)
        return tgtdict


    def _read_parse_procedure(self, file_path, verbose):
        read_ok, file_raw = self._read_file(file_path, verbose)
        parse_ok = True
        tgtdict = None
        msg = None
        if read_ok:
            parse_ok, tgtdict = self._parse_file(file_raw, verbose)
            if parse_ok:
                self.msg("parse ok", verbose)
            else:
                msg = "could not be parsed (parsing error)"
        else:
            msg = "not found"
        return read_ok, parse_ok, msg, tgtdict


    def _read_file(self, file_path, verbose):
        read_ok = False
        file_raw = None
        if os_path_exists(file_path):
            read_ok = True
            self.msg("read ok", verbose)
            with open(file_path, "r") as f:
                file_raw = f.read()
        return read_ok, file_raw


    def _parse_file(self, conf_file_raw, verbose):
        parse_ok = True
        tgtdict = None
        try:
            # rely on tomli's error handling. todo improve
            tgtdict = tomli.loads(conf_file_raw)
        except:
            parse_ok = False
        return parse_ok, tgtdict


    def msg(self, x, v):
        if v:
            print(x)




