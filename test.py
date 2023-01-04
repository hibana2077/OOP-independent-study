class DickError(Exception):
    def __init__(self, msg):
        super().__init__(msg)

raise DickError("雞雞錯亂！！！")