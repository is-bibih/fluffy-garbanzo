class Author():
    """Represents an Author for use in the Bib class."""
    def __init__(self, lname=None, fname=None, mnames=None, is_org=None):
        self.lname = lname
        self.fname = fname
        self.mnames = mnames # should be an array
        self.is_org = is_org

    def __new__(cls, lname=None, fname=None, mnames=None, is_org=None):
        if lname or fname or mnames:
            return object.__new__(cls)
        return None

    def __str__(self):
        string = ''
        if self.lname:
            string += self.lname
        if self.is_org:
            string += '.'
        else:
            if self.fname:
                string += ', ' + self.fname[0]
            if self.mnames:
                for x in self.mnames:
                    string = ' ' + x[0] + '.'
        return string
        