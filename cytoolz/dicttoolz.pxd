cdef dict c_merge(object dicts)


cdef dict c_merge_with(object func, object dicts)


cpdef dict valmap(object func, dict d, object factory=*)


cpdef dict keymap(object func, dict d, object factory=*)


cpdef dict itemmap(object func, dict d, object factory=*)


cpdef dict valfilter(object predicate, dict d, object factory=*)


cpdef dict keyfilter(object predicate, dict d, object factory=*)


cpdef dict itemfilter(object predicate, dict d, object factory=*)


cpdef dict assoc(dict d, object key, object value, object factory=*)


cpdef dict dissoc(dict d, object key)


cpdef dict update_in(dict d, object keys, object func, object default=*, object factory=*)


cpdef object get_in(object keys, object coll, object default=*, object no_default=*)
