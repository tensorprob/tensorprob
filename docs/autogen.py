# -*- coding: utf-8 -*-

# This file has been adapted from https://github.com/fchollet/keras.
# The copyright notice is reproduced here:
# 
# COPYRIGHT
#
# All contributions by François Chollet:
# Copyright (c) 2015, François Chollet.
# All rights reserved.
#
# All contributions by Google:
# Copyright (c) 2015, Google, Inc.
# All rights reserved.
#
# All other contributions:
# Copyright (c) 2015, the respective contributors.
# All rights reserved.
#
# Each contributor holds copyright over their respective contributions.
# The project versioning (Git) records all such contribution source information.
#
# LICENSE
#
# The MIT License (MIT)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import print_function
import re
import inspect
import os
import shutil

from tensorprob import model

MODULES = [(model, 'tensorprob.model'),
          ]

SKIP = [
        'Description',
        'ModelError',
        'current_model',
]
ROOT = 'http://ibab.github.io/tensorprob'
INCLUDE_METHODS_FOR = [
        'Model',
]


def get_earliest_class_that_defined_member(member, cls):
    ancestors = get_classes_ancestors([cls])
    result = None
    for ancestor in ancestors:
        if member in dir(ancestor):
            result = ancestor
    if not result:
        return cls
    return result


def get_classes_ancestors(classes):
    ancestors = []
    for cls in classes:
        ancestors += cls.__bases__
    filtered_ancestors = []
    for ancestor in ancestors:
        if ancestor.__name__ in ['object']:
            continue
        filtered_ancestors.append(ancestor)
    if filtered_ancestors:
        return filtered_ancestors + get_classes_ancestors(filtered_ancestors)
    else:
        return filtered_ancestors


def get_method_signature(method):
    signature = inspect.getargspec(method)
    defaults = signature.defaults
    args = signature.args[1:]
    if defaults:
        kwargs = zip(args[-len(defaults):], defaults)
        args = args[:-len(defaults)]
    else:
        kwargs = []
    st = '%s.%s(' % (method.__module__, method.__name__)
    for a in args:
        st += str(a) + ', '
    for a, v in kwargs:
        if type(v) == str:
            v = '\'' + v + '\''
        elif type(v) == unicode:
            v = 'u\'' + v + '\''
        st += str(a) + '=' + str(v) + ', '
    if kwargs or args:
        return st[:-2] + ')'
    else:
        return st + ')'


def class_to_docs_link(cls):
    module_name = cls.__module__
    assert module_name[:len('tensorprob.')] == 'tensorprob.'
    module_name = module_name[len('tensorprob.'):]
    link = ROOT + module_name.replace('.', '/') + '#' + cls.__name__.lower()
    return link


def class_to_source_link(cls):
    module_name = cls.__module__
    assert module_name[:len('tensorprob.')] == 'tensorprob.'
    path = module_name.replace('.', '/')
    path += '.py'
    try:
        line = inspect.getsourcelines(cls)[-1]
    except OSError:
        line = 0
    link = 'https://github.com/ibab/tensorprob/blob/master/' + path + '#L' + str(line)
    return '[[source]](' + link + ')'


def code_snippet(snippet):
    result = '```python\n'
    result += snippet + '\n'
    result += '```\n'
    return result


def process_class_docstring(docstring):
    docstring = re.sub(r'\n    # (.*)\n',
                       r'\n    __\1__\n\n',
                       docstring)

    docstring = re.sub(r'    ([^\s\\]+):(.*)\n',
                       r'    - __\1__:\2\n',
                       docstring)

    docstring = docstring.replace('    ' * 5, '\t\t')
    docstring = docstring.replace('    ' * 3, '\t')
    docstring = docstring.replace('    ', '')
    return docstring


def process_method_docstring(docstring):
    docstring = re.sub(r'\n        # (.*)\n',
                       r'\n        __\1__\n\n',
                       docstring)

    docstring = re.sub(r'    ([^\s\\]+):(.*)\n',
                       r'    - __\1__:\2\n',
                       docstring)

    docstring = docstring.replace('    ' * 6, '\t\t')
    docstring = docstring.replace('    ' * 4, '\t')
    docstring = docstring.replace('    ', '')
    return docstring

print('Cleaning up existing sources directory.')
if os.path.exists('sources'):
    shutil.rmtree('sources')
print('Populating sources directory with templates.')
for subdir, dirs, fnames in os.walk('templates'):
    for fname in fnames:
        new_subdir = subdir.replace('templates', 'sources')
        if not os.path.exists(new_subdir):
            os.makedirs(new_subdir)
        if fname[-3:] == '.md':
            fpath = os.path.join(subdir, fname)
            new_fpath = fpath.replace('templates', 'sources')
            shutil.copy(fpath, new_fpath)

print('Starting autogeneration.')
covered_so_far = set()
for module, module_name in MODULES:
    class_pages = []
    for name in dir(module):
        if name in SKIP:
            continue
        if name[0] == '_':
            continue
        module_member = getattr(module, name)
        if module_member in covered_so_far:
            continue
        if inspect.isclass(module_member):
            cls = module_member
            if cls.__module__ == module_name:

                try:
                    class_signature = get_method_signature(cls.__init__)
                    class_signature = class_signature.replace('__init__', cls.__name__)
                except:
                    # in case the class inherits from object and does not
                    # define __init__
                    class_signature = module_name + '.' + cls.__name__ + '()'

                methods = []
                methods_not_defined_here = []
                for name in dir(cls):
                    if name in SKIP:
                        continue
                    if name[0] == '_':
                        continue
                    cls_member = getattr(cls, name)
                    if inspect.isfunction(cls_member) or inspect.ismethod(cls_member):
                        method = cls_member
                        signature = inspect.getargspec(method)
                        defaults = signature.defaults
                        args = signature.args[1:]
                        if defaults:
                            kwargs = zip(args[-len(defaults):], defaults)
                            args = args[:-len(defaults)]
                        else:
                            kwargs = []

                        defined_by = get_earliest_class_that_defined_member(method.__name__, cls)
                        if cls == defined_by:
                            methods.append(method)
                        else:
                            methods_not_defined_here.append((method, defined_by))

                blocks = []
                blocks.append('<span style="float:right;">' + class_to_source_link(cls) + '</span>')
                blocks.append('# ' + cls.__name__ + '\n')
                blocks.append(code_snippet(class_signature))
                docstring = cls.__doc__
                if docstring:
                    blocks.append(process_class_docstring(docstring))

                if cls.__name__ in INCLUDE_METHODS_FOR:
                    if methods or methods_not_defined_here:
                        blocks.append('### Methods\n')
                        for method in methods:
                            signature = get_method_signature(method)
                            signature = signature.replace(module_name + '.', '')
                            blocks.append(code_snippet(signature))
                            docstring = method.__doc__
                            if docstring:
                                blocks.append(process_method_docstring(docstring))
                        for method, defined_by in methods_not_defined_here:
                            signature = get_method_signature(method)
                            method_module_name = method.__module__
                            signature = signature.replace(method_module_name + '.', '')
                            link = '[' + defined_by.__name__ + '](' + class_to_docs_link(defined_by) + ')'
                            blocks.append(code_snippet(signature))
                            blocks.append('Defined by ' + link + '.\n')

                mkdown = '\n'.join(blocks)
                class_pages.append((id(cls), mkdown))
                covered_so_far.add(module_member)

    class_pages.sort(key=lambda x: x[0])
    class_pages = [x[1] for x in class_pages]
    module_page = '\n----\n\n'.join(class_pages)

    # save module page.
    # Either insert content into existing page,
    # or create page otherwise
    path = 'sources/' + module_name.replace('.', '/')[len('tensorprob.'):] + '.md'
    if os.path.exists(path):
        template = open(path).read()
        assert '{{autogenerated}}' in template, ('Template found for ' + path +
                                                 ' but missing {{autogenerated}} tag.')
        module_page = template.replace('{{autogenerated}}', module_page)
        print('...inserting autogenerated content into template:', path)
    else:
        print('...creating new page with autogenerated content:', path)
    subdir = os.path.dirname(path)
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    open(path, 'w').write(module_page)
