from bib import Bib

class WebBib(Bib):
    """Bib subclass for Webpages."""
    def __init__(self, url, view_date=None, title=None, auth=None, date=None):
        super(WebBib, self).__init__()
        self.url = url
        self.view_date = view_date # datetime object

    def __str__(self):
        string = ''
        if self.auth:
            string += str(self.auth)
        else:
            string += self.title + '.'
        if self.date:
            string += ' (' + str(self.date) + ').'
        else:
            string += ' (s.f.).'
        if self.title and self.auth:
            string += ' ' + self.title + '.'
        if self.view_date:
            string += ' Recuperado el' + str(self.view_date) + ' de: '
        else:
            string += ' Recuperado de: '
        string += self.url
        return string
