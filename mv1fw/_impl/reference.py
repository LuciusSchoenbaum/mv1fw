



import numpy as np
# todo fw
from torch import from_numpy

from os.path import (
    join as os_path_join,
    exists as os_path_exists,
    dirname as os_path_dirname,
)
from scipy.interpolate import griddata

from .types import (
    get_labels,
    parse_labels,
    get_fslabels,
    parse_fslabels,
)
from .logger import Logger
from .tomlloader import TomlLoader


class Reference:
    """
    Use data and interpolation to generate reference values from input
    time-independent points, and time, or generate
    reference values from a callable method.

    :any:`Reference` is never responsible for emitting files.
    It may access the file system to read, but never to write.

    A goal of the class is to avoid a large, mostly-unused block of data
    in memory. Roughly speaking, it opts instead for
    a smaller "rolling" block of memory that evolves with time t.
    Another goal is flexibility with respect to a choice of requested
    times t, which allows scripts to be updated/modified and enhances
    scalability and testability. Interpolation is applied if a requested t
    is not found in the data.

    If the class has a `callable`, then the dataset is "virtual"
    in the sense that it is generated on the fly from the callable entity,
    instead of read in from storage. In this case, the
    calling convention ``(X, problem)`` is taken from PyPinnch ITCINOOD.
    If you do not us a callable method this does not affect anything.

    Arguments:

        path (string):
            A path to a site where data is located, for a data-defined reference.
            Can be a relative path, if so, it will be a relative to a root path
            defined by the reference intializer (likely, the reference comparison
            target).
        methods:
            A description of analytic methods, for an analytically defined
            reference. If methods is not None, signature: (callable with
            signature X, t, problem -> U). Default: None requires use of
            other arguments (location, dateYMD, steering_handle,
            run_handle) to direct to the data files.
        input_source (string, either 'solution' or 'reference'):
            For a reference data set of the form (inputs, outputs),
            let this data set by denoted (Xref, Yref),
            and let the solver's base slice, and the associated values for the
            given Solution instance, be (Xsolver, Ysolver).
            This parameter determines which side interpolation/evaluation is
            performed on. If 'reference', then the solver is evaluated
            on the reference inputs Xref, which are completely determined by the
            reference (whatever it may be). Then, the Solution instance's method is
            applied to find Ysolver, which may be compared with Yref.
            If 'solution, the reference data Yref is interpolated to the solver's base
            timeslice Xsolver, giving a dataset (Xsolver, Yref_interpolated).
            In this case, the model(s) are never evaluated, at least not for the
            purpose of the reference. Default: 'solution'
        data_format (string):
            Format for data files.
            Basic multi-variable+time format: "mv1". todo document mv1, link
            Only mv1 support ITCINOOD.
        resolution (integer):
            resolution if/when problem.format is called, for filling missing
            input dimensions. Only used if input_source is 'reference',
            and if reference does not provide inputs with the full input
            dimension of the problem.
        verbose (boolean):
            If set, log information about the interpolation that is used when
            data is loaded.

    """

    # todo returning to this I find `input_source` somewhat confusing, review

    def __init__(
            self,
            methods = None,
            path = None,
            data_format = 'mv1',
            input_source = 'solution',
            resolution = 100,
            verbose = False,
    ):
        super().__init__()
        self.times = None
        self.timin = None
        self.timax = None
        # todo only store lbl,indim,with_t and impl fslabels as method
        self.fslabels = None
        self.lbl = None
        self.indim = None
        self.with_t = None
        self.tolerance = None
        self.data_format = data_format
        self.reference_input = input_source != 'solution'
        self.path = path
        self.methods = methods
        if path is not None:
            if methods is not None:
                raise ValueError("Only one of path, methods can be used to define a reference.")
        else:
            if methods is None:
                raise ValueError("Undefined reference. Specify one of path, methods.")
        self.resolution = resolution
        self.verbose = verbose
        self.log = None



    def init(
            self,
            labels,
            root_path = None,
            log = None,
            tolerance = 1e-4,
    ):
        """
        (Not called by user.)

        Arguments:

            labels: (string or list of string)
                labels for input dimensions and output dimensions, separated by commas,
                or a list of individual labels.
                Typical examples:
                Example (0dt): "t; u" (time series)
                Example (1dt): "x, t; u"
                Example (2dt): "x, y, t; u"
            root_path (optional string):
                A root path. If used, it is assumed that (1) the reference
                is data-defined and (2) the
                path set at ``__init__`` was a relative path.
                The full path will be formed using this relative path and the
                ``root_path``. Otherwise, the path set at ``__init__``
                must be a full path.
            tolerance: (float or None)
                If float, interpolate between input slices
                if there is not a time slice within the given tolerance.
                When such interpolation is applied, there will likely be a
                (noticeable) performance cost.
                Note, if this float point tolerance is large enough for
                multiple candidates to fall in the tolerance interval,
                there's no guarantee of taking the best candidate.
                If None, always be content with the closest time slice.
                There may be inaccurate data returned,
                depending on how dense the input slices are,
                however, there is a guarantee to always take the closest time slice
                up to tolerance dt = 1e-10.
            log (optional :any:`Logger`):
                Reference to output log.

        """
        good = False
        # todo run_path predates addition of Location to QueueG, review
        if self.path is not None:
            # > init path
            if root_path is not None:
                self.path = os_path_join(root_path, self.path)
            # todo old approach, delete soon
            # if isinstance(self.path, str):
            #     # > relative path: construct full path as Path instance.
            #     # Assume that a run's path is given by the run_path,
            #     # this means that nothing in that directory has an existence
            #     # independent from the run (i.e., it will be deleted if the job is re-run).
            #     # So the relative directory is interpreted to be relative to a directory
            #     # path one step up from the input run_path, and this is
            #     # informally called the "root_path".
            #     if run_path is not None:
            #         root_path = os_path_dirname(run_path)
            #         self.path = Path(explicit_path=os_path_join(root_path, self.path))
            #     else:
            #         # > absolute path
            #         self.path = Path(explicit_path=self.path)
            # todo review
            if self.data_format == 'mv1':
                # > move into the dat directory
                self.path = os_path_join(self.path, 'dat')
        lbl, indim, with_t = parse_labels(labels)
        self.fslabels = get_fslabels(lbl, indim, with_t)
        self.lbl = lbl
        self.indim = indim
        self.with_t = with_t
        if tolerance is not None and tolerance <= 0.0:
            raise ValueError(f"tolerance {tolerance} must be > 0.0.")
        self.log = log if log is not None else Logger()
        self.tolerance = tolerance
        if with_t:
            if self.path is not None:
                good = self._init_timeseries()
            else:
                # todo review this branch, I'm pretty sure the
                #  assumption I am working under is that in this
                #  branch self.methods is not None.
                pass
        else:
            if self.methods is None:
                # > time-independent and path-based: a single file
                datapath = os_path_join(self.path, f"{self.fslabels}.dat")
                good = os_path_exists(datapath)
                if good:
                    # todo implement data-defined reference time independent problems, this is not so difficult
                    print(f"[Warning] [Reference::init] time-independent Reference is only implemented for use in post-processing.")
        if self.methods is not None:
            # > init methods
            methodsdict = {}
            if callable(self.methods):
                # > there is most likely only one output, but we do not enforce this
                for lb in lbl[indim:]:
                    methodsdict[lb] = self.methods
                self.methods = methodsdict
            elif not isinstance(self.methods, dict):
                raise ValueError(f"Cannot initialize reference methods for reference {labels}.")
            else:
                # > sanity check
                for lb in lbl[indim:]:
                    if lb not in self.methods:
                        raise ValueError(f"No method for computing {lb} given for reference {labels}.")
            # Invariant: methods is a dict, all output labels are in methods.
            good = True
        return good


    ###########################################
    # API:


    def __call__(
            self,
            X = None,
            problem = None,
    ):
        return self.evaluate(X, problem)


    def evaluate(
            self,
            X = None,
            problem = None,
    ):
        """
        Use the dataset or callable to generate output values.
        In symbolic notation, at points X, obtain corresponding values Q.
        Values will be appended to input :any:`XFormat`.

        Arguments:

            X (:any:`XFormat`):
                Target inputs.
            problem (:any:`Problem`):
                Only used if there are callable methods defined.

        """
        Xbase = X.X()
        t = X.t()
        outlbl = self.lbl[self.indim:]
        # Either self.methods or self.path is used.
        if self.methods is not None:
            # > use self.methods
            QQ = np.zeros((Xbase.shape[0], 0))
            for lb in outlbl:
                QQ = np.hstack((QQ, self.methods[lb](X, problem).cpu().detach().numpy()))
        else:
            # > use self.path
            if self.indim == 0:
                tim, tip, found = self._get_timp_timeseries(t)
                if found:
                    QQ = self.times[tim:tim+1,1:]
                else:
                    QQ = self._interpolate_timeseries(t, tim, tip)
            else:
                # > determine whether to interpolate in the higher (with t)
                # or lower (no t) dimension
                tim, tip, found = self._get_timp(t)
                self.msg(f"tim = {tim}, tip = {tip}")
                self.msg(f"tm = {self.times[tim]}, tp = {self.times[tip]}")
                if self.reference_input:
                    # > retrieve inputs, outputs pairs from storage
                    if found:
                        XQQ = self._retrieve_lowdim(tim)
                    else:
                        XQQ = self._retrieve_highdim(t, tim, tip)
                    QQ = XQQ[:,self.indim:]
                else:
                    # > treat data set as evaluator on input X. (use interpolation.)
                    if found:
                        QQ = self._interpolate_lowdim(Xbase, tim)
                    else:
                        QQ = self._interpolate_highdim(Xbase, t, tim, tip)
        # todo use pytorch throughout?
        QQ = from_numpy(QQ)
        # todo update XFormat.append, append Qref as-is
        for i in range(QQ.shape[1]):
            Q = QQ[:,i:i+1]
            lb = f"{outlbl[i]}ref"
            X.append(Q=Q, lb=lb)




    def get_ranges(self):
        """
        From the format file (TOML), obtain the ranges dict.

        Returns:

            dict

        """
        ldr = TomlLoader()
        v = False
        fmtdict = ldr.load(self.path, f"{self.fslabels}.toml", verbose=v)
        ranges = fmtdict['ranges']
        # TOML won't allow None as value.
        for v in ranges:
            if not ranges[v]:
                ranges[v] = None
        return ranges


    def get_times(self, as_array = False):
        """
        Not used in callable case.
        Post-processing utility.

        Return times as a Python list or (optional) as a Numpy array.

        Arguments:

            as_array (boolean): if True, as a numpy array. Default: False

        Returns:

            either a (N,2) shaped float array with rubric [ti, t],
            or a list of pairs (ti, t) of length N.

        """
        # todo review as_array after XFormat
        if self.indim == 0:
            if as_array:
                return self.times[:,0]
            else:
                raise ValueError
        else:
            if as_array:
                out = np.zeros((len(self.times), 2))
                for i, ti in enumerate(self.times):
                    out[i,:] = [float(ti), self.times[ti]]
            else:
                out = []
                for ti in self.times:
                    out.append((ti, self.times[ti]))
            return out


    def get_X(self, t = None):
        """
        Not used in callable case.
        Post-processing utility.

        If data-defined, obtain an X dataset from the data itself.

        Arguments:

            t (scalar): time value

        Returns:

            An array of inputs from data.
            Specifically, an array [x1, x2, ..., xn]
            of inputs where n is known according to the labels.

        """
        if self.indim == 0:
            # > return time series
            return self.times
        else:
            tim, _, found = self._get_timp(t)
            if not found:
                print(f"[Reference] [Warning]  no data found for t = {t}, using data for t = {tim}.")
            fname = os_path_join(self.path, f"{self.fslabels}.t{tim}.dat")
            data = np.loadtxt(fname = fname) # , dtype=real(np)
            return data[:,:self.indim]


    # def reference_exists(self, t = None):
    #     """
    #     Check if dataset XU (inputs and outputs)
    #     exists, with optional parameter t to select a value of t (a timeslice).
    #
    #     Arguments:
    #
    #         t (optional scalar):
    #             target time or `None`, if data is time-independent.
    #
    #     Returns:
    #
    #         boolean
    #
    #     """
    #     if self.indim == 0:
    #         out = True
    #     elif t is None:
    #         fname = os_path_join(self.path, f"{self.fslabels}.dat")
    #         out = os_path_exists(fname)
    #     else:
    #         tim, _, found = self._get_timp(t)
    #         fname = os_path_join(self.path, f"{self.fslabels}.t{tim}.dat")
    #         if not found:
    #             print(f"[Reference] [Warning] no data found for t = {t}, using data for t = {tim}.")
    #         out = os_path_exists(fname)
    #     return out


    def get_XU(self, t = None):
        """
        Not used in callable case.
        Post-processing utility.

        Arguments:

            t (optional scalar):
                target time or `None`, if data is time-independent.

        Returns:

            XU, i.e. all recorded data for timeslice t.
            Specifically, an array of the form [x1, x2, ..., xn, u1, u2, ...um]
            for n inputs and m outputs, where n and m are known
            according to the labels.

        """
        if self.indim == 0:
            data = self.times
        elif t is None:
            fname = os_path_join(self.path, f"{self.fslabels}.dat")
            data = np.loadtxt(fname = fname)
        else:
            tim, _, found = self._get_timp(t)
            fname = os_path_join(self.path, f"{self.fslabels}.t{tim}.dat")
            if not found:
                print(f"[Reference] [Warning] no data found for t = {t}, using data for t = {tim}.")
            data = np.loadtxt(fname = fname)
        # > XFormat conventions: shape is length 2
        if len(data.shape) == 0:
            data = data.reshape([1,1])
        elif len(data.shape) == 1:
            data = data.reshape([-1,data.shape[0]])
        else:
            pass
        return data


    def get_TUs(self):
        """
        Not used in callable case.
        Post-processing utility, for time series.
        cf. :any:`Post`.

        Returns:

            TUs, a list of length m of arrays of the form
            [t, u] for each output u,
            where m is known according to the labels.

        """
        if self.indim == 0 and self.with_t:
            if self.with_t:
                data = self.times
                ts = data[:,0:1]
                TUs = []
                for i in range(1, data.shape[1]):
                    TUs.append(np.hstack((ts, data[:,i:i+1])))
            else:
                raise NotImplementedError
        else:
            raise ValueError
        return TUs


    def get_U(self):
        """
        Return dataset, alias for ``get_XU``.
        Provides a call that is human-readable
        for reference without inputs or time dependence.

        Returns:

            U, i.e. all time-independent recorded data.
            Specifically, an array of the form [u1, u2, ...um]
            for m outputs, where m is known
            according to the labels.

        """
        return self.get_XU(t=None)


    ###########################################
    # private methods:



    def _init_timeseries(self):
        """
        Populate time-dependent values.
        For indim > 0, assumes there exists
        a space-separated value list with lines of the form
            ti, t
        and populates self.times with a dict ti -> t.
        For indim == 0 (time series),
        assumes the data in storage is a time series
        and it loads this in memory as self.times.
        """
        if self.indim == 0:
            # > get the time series
            fname = os_path_join(self.path, f"{self.fslabels}.dat")
            if os_path_exists(fname):
                self.times = np.loadtxt(fname)
            else:
                raise FileExistsError(f"[Reference] file does not exist: {fname}")
        else:
            fname = os_path_join(self.path, f"{self.fslabels}.t.dat")
            self.times = {}
            self.timin = int("FFFFFFFF", 16)
            self.timax = 0
            if os_path_exists(fname):
                with open(fname) as f:
                    lines = f.readlines()
                    lines = [x.strip() for x in lines]
                    for line in lines:
                        svals = line.split(" ")
                        ti = int(svals[0])
                        t = float(svals[1])
                        self.times[ti] = t
                        if ti > self.timax:
                            self.timax = ti
                        if ti < self.timin:
                            self.timin = ti
            else:
                raise FileExistsError(f"[Reference] file does not exist: {fname}")
        # todo return value obsolesced, review
        return True

    def _get_timp_timeseries(self, t):
        """
        For time series take tolerance = 0.0 (because
        interpolating when t not found is cheap),
        and use binary search instead of slower brute force
        because there may be many time values to search.

        :param t:
        :return: tim, tip, found
        """
        TQQ = self.times
        tim = 0
        tip = TQQ.shape[0]-1
        tih = int((tip - tim)/2.0)
        found = False
        if t == TQQ[tip,0]:
            found = True
            tim = tip
        if t == TQQ[tim,0]:
            found = True
            tip = tim
        if not found:
            while tih != tim:
                ttest = TQQ[tih,0]
                if t == ttest:
                    tim = tih
                    tip = tih
                    found = True
                    break
                elif t < ttest:
                    tip = tih
                else:
                    tim = tih
                tih = int((tip - tim)/2.0) + tim
        # Invariant: if found, tip = tim.
        # Invariant: if not found, tip - tim = 1 and t is properly inside the captured interval.
        return tim, tip, found


    def _interpolate_timeseries(self, t, tim, tip):
        # > linear interpolate
        t0 = self.times[tim,0]
        t1 = self.times[tip,0]
        qq0 = self.times[tim,1:]
        qq1 = self.times[tip,1:]
        Qref = np.zeros((1, self.times.shape[1]-1))
        s = (t - t0)/(t1-t0)
        Qref[0,:] = qq0 + s*(qq1-qq0)
        return Qref


    def _get_timp(self, t):
        """
        This is only called once per
        __call__, so we search by brute force.

        Arguments:

            t (scalar):
                requested time, interpolation target

        Returns:

            tim, tip, found (integer, integer, boolean):
                tim <= tip

        """
        tim = self.timin
        tm = self.times[tim]
        tip = self.timax
        tp = self.times[tip]
        found = False
        if self.tolerance is not None:
            # > use requested tolerance.
            # Remind: if this tolerance is large,
            # there's no guarantee of taking the best candidate.
            tol = self.tolerance
        else:
            # > first find tip, tim using a small tolerance.
            tol = 1e-10
        if t < tm - tol:
            raise ValueError(f"t value {t} out of range for interpolation: tmin = {tm}, tmax = {tp}.")
        if t > tp + tol:
            raise ValueError(f"t value {t} out of range for interpolation: tmin = {tm}, tmax = {tp}.")
        for txi in self.times:
            tx = self.times[txi]
            if -tol < t - tx < +tol:
                tim = txi
                tip = txi
                found = True
                break
            else:
                if tx < t:
                    if tx > tm:
                        # raise interval
                        tim = txi
                        tm = tx
                else: # t_ > t:
                    if tx < tp:
                        # lower interval
                        tip = txi
                        tp = tx
        if self.tolerance is None and not found:
            # > take the nearest of the two obtained
            ti_nearest = tim
            if t - tm > tp - t:
                ti_nearest = tip
            tim = ti_nearest
            tip = ti_nearest
            found = True
        return tim, tip, found



    def _retrieve_lowdim(
            self,
            tiref,
    ):
        """
        todo document
        """
        fname = os_path_join(self.path, f"{self.fslabels}.t{tiref}.dat")
        self.msg(f"retrieve low dim\nfname = {fname}")
        data = np.loadtxt(fname = fname) # , dtype=real(np)
        XQ = data
        return XQ



    def _retrieve_highdim(
            self,
            t,
            tim,
            tip,
    ):
        """
        todo document
        """
        # (RetrieveBranch)
        # This branch is the third and last, to be avoided if possible.
        indim = self.indim
        tm = self.times[tim]
        tp = self.times[tip]
        fnamem = os_path_join(self.path, f"{self.fslabels}.t{tim}.dat")
        fnamep = os_path_join(self.path, f"{self.fslabels}.t{tip}.dat")
        self.msg(f"interpolate high dim\nfname = {fnamem}\nfname = {fnamep}")
        datam = np.loadtxt(fname = fnamem) # , dtype=real(np)
        datap = np.loadtxt(fname = fnamep) # , dtype=real(np)
        # > build points tuple
        points = []
        for i in range(indim):
            Xrefi = np.vstack((datam[:,i:i+1], datap[:,i:i+1]))
            points.append(Xrefi[:,0])
        Xreft = np.vstack((np.full((datam.shape[0],1), tm),np.full((datap.shape[0],1), tp)))
        points.append(Xreft[:,0])
        # > interpolate to the points in datam, shifted to time t
        X = np.hstack((datam[:,:indim], np.full((datam.shape[0],1), t)))
        # > build XQ by stacking Q columns on time-independent X
        XQ = np.copy(datam[:,:indim])
        for out1 in range(indim, len(self.lbl)):
            Uref = np.vstack((datam[:,out1:out1+1], datap[:,out1:out1+1]))
            U = griddata(
                points=tuple(points),
                values=Uref,
                xi=X,
                method="linear",
            )
            XQ = np.hstack((XQ, U))
        return XQ



    def _interpolate_lowdim(
            self,
            X,
            tiref,
    ):
        fname = os_path_join(self.path, f"{self.fslabels}.t{tiref}.dat")
        self.msg(f"interpolate low dim\nfname = {fname}")
        data = np.loadtxt(fname = fname) # , dtype=real(np)
        points = []
        for i in range(self.indim):
            Xrefi = data[:,i]
            points.append(Xrefi)
        UU = np.zeros((X.shape[0],0))
        for out1 in range(self.indim, len(self.lbl)):
            Uref = data[:,out1]
            U = griddata(
                points=tuple(points),
                values=Uref,
                xi=X[:,:self.indim].cpu().detach().numpy(),
                method="nearest", # avoids issues with convex hull
            ).reshape((X.shape[0],1))
            UU = np.hstack((UU, U))
        return UU


    def _interpolate_highdim(
            self,
            X,
            t,
            tim,
            tip,
    ):
        tm = self.times[tim]
        tp = self.times[tip]
        fnamem = os_path_join(self.path, f"{self.fslabels}.t{tim}.dat")
        fnamep = os_path_join(self.path, f"{self.fslabels}.t{tip}.dat")
        self.msg(f"interpolate high dim\nfname = {fnamem}\nfname = {fnamep}")
        datam = np.loadtxt(fname = fnamem) # , dtype=real(np)
        datap = np.loadtxt(fname = fnamep) # , dtype=real(np)
        # > build points tuple
        points = []
        for i in range(self.indim):
            Xrefi = np.vstack((datam[:,i:i+1], datap[:,i:i+1]))
            points.append(Xrefi[:,0])
        Xreft = np.vstack((np.full((datam.shape[0],1), tm),np.full((datap.shape[0],1), tp)))
        points.append(Xreft[:,0])
        UU = np.zeros((X.shape[0],0))
        # > higher-dimension interpolation between two slices: this may be costly
        for out1 in range(self.indim, len(self.lbl)):
            Uref = np.vstack((datam[:,out1:out1+1], datap[:,out1:out1+1]))
            # > the lower timeslice shifted to the target time t
            Xi = np.hstack((datam[:,0:self.indim], np.full((datam.shape[0],1), t)))
            # > the values of the target on the new timeslice Xi
            Uref = griddata(
                points=tuple(points),
                values=Uref,
                xi=Xi,
                # the idea is to average the upper and lower slice but the input points on the
                # two slices are not assumed to be the same (the grid can be time-adaptive)
                # so a general interpolation (griddata) is used.
                method="linear",
            )
            # > munge for scipy
            points1 = []
            for i in range(self.indim):
                points1.append(Xi[:,i])
            # > interpolate Xi to the target points X
            U = griddata(
                points=tuple(points1), # shape [m,], number: indim
                values=Uref, # shape [m,]
                xi=X[:,:self.indim].cpu().detach().numpy(), # shape [n, indim]
                method="nearest", # avoids issues with convex hull
            ) # shape [n,]
            U = U.reshape((X.shape[0],1))
            UU = np.hstack((UU, U))
        return UU


    def msg(self, x):
        if self.verbose:
            self.log(x)


    def __str__(self):
        out = ""
        # draft
        if self.methods is not None:
            typ = "Method-Defined"
        else:
            typ = "Data-Defined"
        out += f"{typ}: {get_labels(*parse_fslabels(self.fslabels))}\n"
        return out


