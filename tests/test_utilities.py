from tensorprob import utilities


def test_generate_name():
    class SomeTestClass(object):
        pass

    def some_test_function():
        pass

    assert utilities.generate_name(SomeTestClass) == 'SomeTestClass_1'
    assert utilities.generate_name(some_test_function) == 'some_test_function_1'
    assert utilities.generate_name(SomeTestClass) == 'SomeTestClass_2'
    assert utilities.generate_name(some_test_function) == 'some_test_function_2'
    assert utilities.generate_name(some_test_function) == 'some_test_function_3'


def test_class_property():
    class SomeTestClass(object):
        @utilities.classproperty
        def new_class_property(cls):
            return cls, 'test_string'

    assert SomeTestClass.new_class_property == (SomeTestClass, 'test_string')


def test_grouper():
    grouped = utilities.grouper([0, 1, 2, 3, 4], n=2)
    assert list(grouped) == [(0, 1), (2, 3), (4, None)]
