import pyro

from .poutine import Poutine
from .lambda_poutine import LambdaPoutine


class BlockPoutine(Poutine):
    """
    Blocks some things
    """

    def __init__(self, fn,
                 hide_all=True, expose_all=False,
                 hide=None, expose=None,
                 hide_types=None, expose_types=None):
        """
        Constructor for blocking poutine
        Default behavior: block everything
        """
        super(BlockPoutine, self).__init__(fn)
        # first, some sanity checks:
        # hide_all and expose_all intersect?
        assert (hide_all is False and expose_all is False) or \
               (hide_all != expose_all), "cannot hide and expose a site"

        # hide and expose intersect?
        if hide is None:
            hide = []
        else:
            hide_all = False

        if expose is None:
            expose = []
        assert set(hide).isdisjoint(set(expose)), \
            "cannot hide and expose a site"

        # hide_types and expose_types intersect?
        if hide_types is None:
            hide_types = []
        if expose_types is None:
            expose_types = []
        assert set(hide_types).isdisjoint(set(expose_types)), \
            "cannot hide and expose a site type"

        # now set stuff
        self.hide_all = hide_all
        self.expose_all = expose_all
        self.hide = hide
        self.expose = expose
        self.hide_types = hide_types
        self.expose_types = expose_types

    def _block_up(self, msg):
        """
        A stack-blocking operation
        """
        # decision rule for hiding:
        if (msg["name"] in self.hide) or \
           (msg["type"] in self.hide_types) or \
           ((msg["name"] not in self.expose) and
            (msg["type"] not in self.expose_types) and self.hide_all):  # noqa: E129

            return True
        # otherwise expose
        else:
            return False

    def _pyro_sample(self, msg):
        ret = super(BlockPoutine, self)._pyro_sample(msg)
        msg.update({"stop": self._block_up(msg)})
        return ret

    def _pyro_observe(self, msg):
        ret = super(BlockPoutine, self)._pyro_observe(msg)
        msg.update({"stop": self._block_up(msg)})
        return ret

    def _pyro_map_data(self, msg):
        name, data, fn, batch_size, batch_dim = \
            msg["name"], msg["data"], msg["fn"], msg["batch_size"], msg["batch_dim"]
        scale = pyro.util.get_batch_scale(data, batch_size, batch_dim)
        msg.update({"fn": LambdaPoutine(fn, name, scale)})
        ret = super(BlockPoutine, self)._pyro_map_data(msg)
        msg.update({"fn": fn})
        msg.update({"stop": self._block_up(msg)})
        return ret

    def _pyro_param(self, msg):
        ret = super(BlockPoutine, self)._pyro_param(msg)
        msg.update({"stop": self._block_up(msg)})
        return ret
