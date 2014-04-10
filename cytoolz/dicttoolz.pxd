cdef dict c_merge(object dicts)


cdef dict c_merge_with(object func, object dicts)


cpdef dict valmap(object func, object d)


cpdef dict keymap(object func, object d)


cpdef dict valfilter(object predicate, object d)


cpdef dict keyfilter(object predicate, object d)


cpdef dict assoc(object d, object key, object value)


cpdef dict update_in(object d, object keys, object func, object default=*)


cpdef object get_in(object keys, object coll, object default=*, object no_default=*)
