from .register import Register


FS = Register('FULLSUPERVISED')
FS.add_children(Register('MODELS'))

