


from math import log2, log10




# helper
# parse labels <inputs, csv> ; <outputs, csv>
# or <inputs, csv>
def parse_labels(labels):
    with_t = False
    ti = 0
    lbl0 = labels.split(';')
    if len(lbl0) == 0:
        raise ValueError(f"Error parsing labels {labels}. Separate inputs and outputs with a semicolon (;).")
    else:
        lbl = [x.strip() for x in lbl0[0].split(',')]
        for i, lb in enumerate(lbl):
            if lb == 't':
                with_t = True
                ti = i
        if with_t:
            lbl = lbl[:ti]+lbl[ti+1:]
        indim = len(lbl)
        if len(lbl0) > 1:
            lblout = [x.strip() for x in lbl0[1].split(',')]
            lbl += lblout
    return lbl, indim, with_t

# helper
# the same output as parse_labels, from an fslabels input.
def parse_fslabels(fslabels):
    # Note: split() places an empty set if the pattern
    # is found on the edge of the string.
    lbl0 = fslabels.split('--')
    if len(lbl0) != 2:
        raise ValueError(f"Error parsing fslabels {fslabels}.")
    else:
        with_t = False
        lbl = lbl0[0].split('-')
        if lbl[-1] == 't':
            with_t = True
            lbl = lbl[:-1]
        indim = len(lbl)
        if lbl0[1] != '':
            outlbl = lbl0[1].split('-')
            lbl += outlbl
    return lbl, indim, with_t


# helper
def get_labels(lbl, indim, with_t):
    lbl0 = lbl[:indim]
    if with_t:
        lbl0.append('t')
    out = ", ".join(lbl0)
    if indim < len(lbl):
        out += '; ' + ", ".join(lbl[indim:])
    return out


# helper
# from labels = "x, y; u" make "x-y--u" etc.
# fs for "file system"
# Note: fslabels is required/specified to have exactly one substring '--'.
def get_fslabels(lbl, indim, with_t):
    lblin = lbl[:indim]
    if with_t:
        lblin += 't'
    out = "-".join(lblin)
    out += '--'
    if indim < len(lbl):
        out += "-".join(lbl[indim:])
    return out





# helper
def popcount(j):
    """
    Get the popcount, or number of set bits.

    There is int.bit_count, but only for
    some versions of Python 3.
    """
    out = 0
    x = j
    while x > 0:
        x &= x - 1
        out += 1
    return out



# helper
def get_bit(N, i):
    """
    Get the ith bit of N
    """
    return (N >> i) & 1




# helper
# remove the path, for example,
# from "data/u_" get "u_", assuming Unix directory formatting.
def split_path(path):
    flist = path.split("/")
    return "/".join(flist[:-1]), flist[-1]


# helper
# count rightmost zeros in int n up to position M
def rightzeros(n, M):
    m, N = 0, n
    while N % 2 == 0:
        N = N >> 1
        m = m+1
        if m == M:
            break
    return m


# helper
# whether n is a power of 2 or not
def ispow2(n):
    if n <= 0:
        return False
    if n > 2**(int(log2(n))):
        return False
    else:
        return True



# helper
# number of digits in base 10 form
def width10(x):
    return 1 if x == 0 else int(log10(x))+1



# helper
def tag_filename(filename, insert):
    """
    Tag a filename after it has already been constructed.
    Useful when multiple filenames are needed that
    relate to one another.

    Arguments:

        filename (string):
            a filename.
        insert (string):
            a tag to insert, before the ending.

    """
    dotsplit = filename.split(".")
    out = dotsplit[0]
    if len(dotsplit) > 1:
        for piece in range(1,len(dotsplit)-1):
            out += "." + dotsplit[piece]
        out += "." + insert + "." + dotsplit[-1]
    else:
        out += "." + insert
    return out



# helper
def tag_filename_front(filename, insert):
    """
    Tag a filename after it has already been constructed.
    Useful when multiple filenames are needed that
    relate to one another.

    Arguments:

        filename (string):
            a filename.
        insert (string):
            a tag to insert, at front of name.

    """
    lst = filename.split("/")
    name = insert + "." + lst[-1]
    return "/".join(lst[:len(lst)-1] + [name])



# helper
def smallest_nonzero(x, hint = 0):
    if x == 0:
        return -1
    i = hint
    n = x >> hint
    safety = 0
    while n & 1 == 0:
        n = n >> 1
        i += 1
        if safety == 64:
            break
        else:
            safety += 1
    return i


# helper
def unset_at_index(x, i):
    n = x
    mask = ~(1 << i)
    n = n & mask
    return n


# helper
# whether exactly one is not None, no tricks applied
def xor_None(a, b):
    A = a is None
    B = b is None
    return (A and not B) or (B and not A)



# helper
def sortdown(X, k = 0, row = False):
    """
    Sort a two-axis array by a choice of row or column.

    Arguments:

        X:
            Target array
        k (integer):
            index of row or column. Default: 0
        row (boolean):
            If True, sort by row k. Default: False (sort by column k).

    Returns:

        array
    """
    if row:
        return X[:,X[k,:].argsort()]
    return X[X[:,k].argsort(),:]


# helper
def tag(stringin):
    """
    Convert a string representing an ordinary/common floating
    point value into a file system-safe tag.

    Arguments:

        stringin (string)

    Returns:

        string

    """
    badchars = "-."
    badmap = {
        '-': 'm',
        '.': 'p',
    }
    out = ""
    for char in stringin:
        for bchar in badchars:
            if char == bchar:
               out += badmap[char]
            else:
                out += char
    return out




