import author

class Bib():
    """Base class for citations. Should not be instantiated directly."""
    def __init__(self, title=None, auth=None, date=None, is_digital=None):
        self.title = title
        self.auth = auth #can be an array for multiple Author objects
        self.date = date #datetime
        self.is_digital = is_digital #should be set to None for some subclasses
