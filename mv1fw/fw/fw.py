


from numpy import float32, float64

framework = "torch"

if framework == "torch":
    try:
        # data types
        from torch import \
            float32 as fw_float32, \
            float64 as fw_float64
        # backend/gpu utilities
        from torch import \
            set_default_dtype as fw_set_default_dtype, \
            manual_seed as fw_manual_seed
        from torch.cuda import \
            is_available as fw_cuda_is_available, \
            device_count as fw_cuda_device_count, \
            get_device_name as fw_cuda_get_device_name, \
            manual_seed_all as fw_cuda_manual_seed_all
        from torch.backends.mps import \
            is_available as fw_backends_mps_is_available, \
            is_built as fw_backends_mps_is_built
        # tensor arithmetic/algebra
        from torch import \
            tensor as fw_Tensor, \
            from_numpy as fw_from_numpy, \
            hstack as fw_hstack, \
            vstack as fw_vstack, \
            zeros as fw_zeros, \
            full as fw_full, \
            randperm as fw_randperm
    except ImportError:
        # > fall back to Numpy
        # todo
        from numpy import \
            float32 as fw_float32, \
            float64 as fw_float64




# pytorch reduction to MSE:
# Reduce = torch.nn.MSELoss(reduction="mean") # L2Loss
# mse = Reduce(input=v_pred, target=v_true)



# helper
def cpu(backend):
    return backend == "cpu"


# helper
def cuda_if_available():
    return "cuda" if fw_cuda_is_available() else "cpu"


# helper
# obtain numpy-style shape from tensor x
def fw_shape(x):
    return list(x.shape)


dtype_dict = {float32: fw_float32, float64: fw_float64}


# helper
# convert dtype to equivalent framework type or "fw_type".
# todo review
def fw_type(dtype):
    out = dtype_dict[dtype] if dtype in dtype_dict else None
    if out is None:
        raise ValueError(f"Unrecognized dtype {dtype}")
    return out


# helper
# convert a fw_type (framework dtype) back to a dtype (numpy dtype)
def get_dtype(fw_type):
    out = None
    for dtype in dtype_dict:
        fw_type_ = dtype_dict[dtype]
        if fw_type_ == fw_type:
            out = dtype
    if out is None:
        raise ValueError(f"Unrecognized fw_type {fw_type}")
    return out


# helper
# a scalar with shape compatible with torch + this library's conventions
def fw_scalar(dtype, a = 0.0, requires_grad = False):
    return fw_full(
        size=[1,1],
        fill_value=a,
        dtype=dtype,
        requires_grad=requires_grad
    )


# helper
# print the dtype, type(), and host device of a tensor.
def dtypecheck(name, t):
    if not isinstance(t, fw_Tensor):
        print(f"type {name}: {type(t)}")
    else:
        print(f"dtype {name}: {t.dtype} {t.type()} {t.device}")





class XNoF:
    """
    Wrapping class that
    abstracts the interface with the
    tensor library framework used.


    Parameters:

        X (optional tensor):
            tensor to initialize
        shape (optional pair of integers):
            shape to initialize empty
        reserve (optional integer):
            reserved space for variables (Default: 0)

    """

    # todo change name to NoFormat?

    def __init__(
            self,
            X = None,
            shape = None,
            reserve = 0,
    ):
        if X is None:
            if shape is None:
                raise ValueError
            # todo send to device, requires grad, dtype ....
            self._X = fw_zeros(shape)
        else:
            if X is None:
                raise ValueError
            # todo if X is Numpy, swing it over to the framework
            self._X = X
        if reserve < 0:
            raise ValueError
        else:
            self.reserve = reserve
        # Invariant: X returns a tensor object.
        # Invariant: self.reserve is an integer >= 0.

        self.n = self._X.shape[1]
        # > set up reserved space
        N = self._X.shape[0]
        self._X = fw_hstack((self._X, fw_zeros((N,self.reserve)).to(X.device)))
        # memoizer for making pitstops
        self._former_device = None
        self._former_requires_grad = None


    def X(self):
        """
        Extract the underlying tensor.

        """
        return self._X


    def enter_pitstop(self):
        """
        A pitstop is an abstraction
        that wraps processing on the cpu.
        It is (by definition) an entrance
        to an environment that the tensor
        will leave again using exit_pitstop,
        within the same control flow.

        This mechanism can be used, for example,
        for plotting routines.

        Most likely, the code for a pitstop
        moves the tensor to the cpu,
        and into the numpy environment,
        but this condition is not a requirement.

        """
        # these are all the attributes of a torch tensor ITCINOOD.
        # self._former_dtype = self._X.dtype
        self._former_device = self._X.device
        self._former_requires_grad = self._X.requires_grad
        # self._former_layout = self._X.layout
        # self._former_memory_format = self._X.memory_layout
        self._X = self._X.cpu().detach().numpy()


    def exit_pitstop(self):
        """
        Restore the state of the data
        prior to entering the pit stop.

        """
        # dtype? layout?
        # improve. I bundled the procedure
        # here so we can refine it later, that is PRECISELY the point!
        self._X = fw_from_numpy(self._X) \
            .to(device=self._former_device)
        if not self._former_requires_grad:
            self._X.detach()
        self._former_device = None
        self._former_requires_grad = None
        # Note: I don't know why the
        # torch documentation does not
        # state whether a from_numpy tensor
        # requires grad.


    def bulk(self):
        """
        Memory occupied with information,
        ignoring the size.
        """
        return self.n


    def size(self):
        """
        Number of entries/points/etc. available/present.
        """
        return self._X.shape[0]


    def cosize(self):
        """
        Actual memory available for appending,
        including the reserve appending space.
        """
        return self._X.shape[1]


    def reset(self, n_restore = None):
        """
        Restore to a previous state, after appending.

        Arguments:

            n_restore (optional integer):
                Restore the bulk back to this amount,
                leaving the remainder in reserve.

        """
        if n_restore is not None:
            self.n = n_restore
        self.reserve = self._X.shape[1] - self.n


    def append(self, Q):
        """
        Extend along the bulk via a
        quantity or quantities Q.

        Arguments:

            Q (torch tensor):
                to append.

        """
        nQ = Q.shape[1]
        if nQ <= self.reserve:
            n = self.n
            self._X[:,n:n+nQ] = Q
            self.reserve -= nQ
            self.n += nQ
        else:
            # zero reserve
            self._X = fw_hstack((self._X, Q))
            self.n += nQ



    def extend(self, X2):
        """
        Extend along the size with
        a set of points/values/etc. X2.

        Arguments:

            X2 (torch tensor):
                to extend.

        """
        self._X = fw_vstack((self._X, X2))


    def shuffle(self):
        idxs = fw_randperm(self._X.shape[0])
        return self._X[idxs]


