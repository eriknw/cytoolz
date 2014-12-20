cdef dict c_merge(object dicts)


cdef dict c_merge_with(object func, object dicts)


cpdef dict valmap(object func, dict d)


cpdef dict keymap(object func, dict d)


cpdef dict itemmap(object func, dict d)


cpdef dict valfilter(object predicate, dict d)


cpdef dict keyfilter(object predicate, dict d)


cpdef dict itemfilter(object predicate, dict d)


cpdef dict assoc(dict d, object key, object value)


cpdef dict dissoc(dict d, object key)


cpdef dict update_in(dict d, object keys, object func, object default=*)


cpdef object get_in(object keys, object coll, object default=*, object no_default=*)
