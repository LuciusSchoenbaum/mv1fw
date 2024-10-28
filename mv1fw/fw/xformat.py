

from .._impl.types import (
    parse_labels,
    get_fslabels,
    parse_fslabels,
)

from .fw import XNoF



class XFormat(XNoF):
    """
    Formatted data structure for housing
    an array with optional labels.
    Designed specifically to be processed
    by :any:`Problem` in the get() method.

    The user should not need to see or be aware
    of :any:`XFormat`. It arises only as a symbol ``X``
    in the argument of :any:`Problem.get` and
    in arguments in the problem description,
    and it is passed as-is to the get() method
    and the get_moment() method.

    A motivation for :any:`XFormat`
    is that data being swapped between classes
    is a timeslice far more often than a 'slice' of any other kind.
    A general structured data type that allows for
    the possibility of a timeslice as well as the possibility
    of a time-dependent set of data, is convenient.

    .. note::

        If ``labels`` is None, then the formatting
        is assumped to be the same as that given in the
        problem description; the only possible exception
        is the timeslice case (see next Note).
        However, if ``labels`` is not None, this does
        not exclude that possibility.

    .. note::

        If t() produces None, then the time is expressed
        in the array ``X`` at the column ``indim``.
        If not, then ``X`` lacks a time column, and the column
        after ``indim-1``, if any, is an output column.
        Thus the t() method is all you need to test
        whether or not the data structure is a timeslice.

    Arguments:

        X (:any:`XNoF`):
            input data. Even if there is no data "yet",
            caller must provide X tensor in order to
            set the device, dtype, etc.
        t (optional scalar):
            time correlated to X, in the common case that
            X lives on a time slice.
        labels (optional string):
            labels for X, including t if time is included either
            as part of X or as t.
        fslabels (optional string):
            An fslabels object. If one is available, use this argument
            instead of labels to avoid a parsing step. STIUYKB.
        reserve (integer):
            If used, reserve this many additional unlabeled columns
            to be set via :any:`XFormat.append` .
            This isn't required in order to use :any:`XFormat.append`
            but it may relieve pressure on memory allocation.
            (Default: 0)

    """

    def __init__(
            self,
            X,
            t = None,
            labels = None,
            fslabels = None,
            reserve = 0,
    ):
        if t is not None and not isinstance(t, float):
            raise ValueError(f"time t must be either float or None.")
        # todo closing off this possibility (during early stages), but consider allowing it
        if X is None:
            raise ValueError
        if X is None:
            if t is None:
                raise NotImplementedError
            else:
                # > arguably, it can be interpreted as a timeslice for 0d inputs.
                #  for now, however, require that X is given and review later.
                self._fslabels = "t"
        if labels is None and fslabels is None:
            # Require: a format is defined at init.
            raise ValueError(f"A format is required at init, in order to define XFormat object.")
        else:
            if fslabels:
                self._fslabels = fslabels
            else:
                self._fslabels = get_fslabels(*parse_labels(labels))
        super().__init__(
            X=X,
            reserve=reserve,
        )
        self._t = t


    def t(self):
        return self._t


    def fslabels(self):
        return self._fslabels


    def reset(self, n_restore = None):
        """
        Clear the output data.
        The input data (incl. time) will not change.

        This is not a deep operation (affects formatting only.)

       Arguments:

            n_restore (optional integer):
                Restore the bulk back to this amount,
                leaving the remainder in reserve.

        """
        lbl, indim, with_t = parse_fslabels(self._fslabels)
        if n_restore is not None:
            if n_restore < indim:
                indim_ = n_restore
            else:
                indim_ = indim
            n_restore_ = n_restore
        else:
            indim_ = indim
            n_restore_ = indim
        # > update labels
        self._fslabels = get_fslabels(lbl[:n_restore_], indim_, with_t)
        # > update data
        super().reset(n_restore=n_restore_)


    def advance(self, deltat):
        """
        Advance the inputs in time.
        this corrupts the output data,
        so it will be cleared.

        """
        self.reset()
        if self._t is None:
            # There are two possibilities:
            # (1) the data is time independent,
            # or (2) the data is time-dependent and time is
            # heterogeneous. We perform the operation
            # assuming case (1) for the time being.
            pass
        else:
            self._t += deltat



    def append(self, Q, lb):
        """
        Append data Q with label lb.
        Q must have the same point size as the input data.

        Arguments:

            Q (Nx1 pytorch tensor):
            lb (string):
                label for Q

        """
        # todo multiple label append
        lbl, indim, with_t = parse_fslabels(self._fslabels)
        lbl.append(lb)
        self._fslabels = get_fslabels(lbl, indim, with_t)
        super().append(Q)




