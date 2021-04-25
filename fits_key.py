"""Supplements to astropy.nddata
"""
import re
import warnings

import astropy.units as u
from astropy import log
from astropy.io import fits
from astropy.io.fits.card import Card, UNDEFINED
from astropy.nddata import NDArithmeticMixin

# Create the unit regular expression string from configurable pieces.
# Below implements r'\ \(.*\)$' which matches " (<unit>)" at the end
# of the FITS card comment

####### PR1 ########
class FitsKeyQuantityHeader(fits.Header):
    # --> This would eventually be a conf item
    return_key_as_quantity = 'never'
    # --> These could be conf items or standardized
    unit_str_start = ' '
    unit_str_delimeters = '()'
    unit_str_end = ''
    unit_str_position = 'end'
    
    """
    """
    def __init__(self,
             *args,
             return_key_as_quantity=None,
             unit_str_start=None,
             unit_str_delimeters=None,
             unit_str_end=None,
             unit_str_position=None,
             **kwargs):
        super().__init__(*args, **kwargs)
        self.return_key_as_quantity = return_key_as_quantity or self.return_key_as_quantity
        self.unit_str_start = unit_str_start or self.unit_str_start
        self.unit_str_delimeters = unit_str_delimeters or self.unit_str_delimeters
        self.unit_str_end = unit_str_end or self.unit_str_end
        self.unit_str_position = unit_str_position or self.unit_str_position
        
        
    @property
    def _unit_regexp(self):
        s = re.escape(self.unit_str_start)
        l = re.escape(self.unit_str_delimeters[0])
        r = re.escape(self.unit_str_delimeters[1])
        e = re.escape(self.unit_str_end)
        unit_str_position = self.unit_str_position.lower()
        if unit_str_position in ['start', 'beginning']:
            ms = '^'
            me = ''
        elif unit_str_position == 'end':
            ms = ''
            me = '$'
        else:
            raise ValueError(f'Unknown unit string position {self.unit_str_position}.  Expecting "start" or "end"')
        return f'{ms}{s}{l}.*{r}{e}{me}'
        

    def get_fits_key_unit(self, key):
        """Returns `~astropy.units.Unit` of card if defined, else ``None``"""
        #if self.get(key) is None:
        #    raise KeyError(f'No key "{key}" found')
        kcomment = self.comments[key]
        # Extract unit together with lead-in and delimiter
        # https://stackoverflow.com/questions/8569201/get-the-string-within-brackets-in-python
        m = re.search(self._unit_regexp, kcomment)
        if m is None:
            log.debug(f'no unit matching "(unit)" in "{kcomment}"')
            return None
        # Strip off delemeters
        punit_str = m.group(0)
        # No escaping needed because within re '[]'
        s = self.unit_str_start
        l = self.unit_str_delimeters[0]
        r = self.unit_str_delimeters[1]
        e = self.unit_str_end
        unit_str = re.sub(f'[{s}{l}{r}{e}]', '', punit_str)
        try:
            unit = u.Unit(unit_str)
        except ValueError as e:
            log.warning(f'Card comment: {kcomment}')
            log.warning(e)
            return None
        return unit

    def del_fits_key_unit(self, key):
        """Deletes `~astropy.units.Unit` of card"""
        if self.get_fits_key_unit(key) is None:
            raise ValueError(f'No unit for key "{key}" to delete')
        kcomment = self.comments[key]
        print(kcomment)
        print(self._unit_regexp)
        kcomment = re.sub(self._unit_regexp, '', kcomment)
        self.comments[key] = kcomment

    def set_fits_key_unit(self, key, unit):
        """Sets `~astropy.units.Unit` of card"""
        value = self.get(key)
        if value is None:
            raise KeyError('No key "{key}" found')
        if not isinstance(unit, u.UnitBase):
            raise ValueError('unit is not an instance of astropy.units.Unit')
        try:
            self.del_fits_key_unit(key)
        except ValueError:
            pass
        kcomment = self.comments[key]
        unit_str = unit.to_string()
        # Calculate how much room we need for the unit on the end of
        # kcomment so we can truncate the comment if necessary.  Doing it
        # this way lets astropy handle the HEIRARCH stuff, which moves the
        # start column of the comment
        uroom = len(unit_str) + 1    
        # Use raw Card object to calculate the card image length to
        # make sure our unit will fit in.  Ignore warning about converting
        # cards to HEIRARCH
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=fits.verify.VerifyWarning)
            c = Card(key, value, kcomment)
        im = c.image
        # Find how many spaces are used to pad the comment to 80
        m = re.search(' *$', im)
        if m is None:
            num_spaces = 0
        else:
            num_spaces = len(m.group(0))
        shorten = max(0, uroom - num_spaces)
        kcomment = kcomment[0:len(kcomment)-shorten]
        kcomment = f'{kcomment} ({unit_str})'
        self.comments[key] = kcomment

    def get_fits_key_quantity(self, key, return_key_as_quantity=None):
        """Gets `~astropy.units.Quantity` of card"""
        rkaq = return_key_as_quantity or self.return_key_as_quantity
        if rkaq not in ['never', 'recognized', 'always']:
            raise ValueError(f'Unrecognized value for "return_key_as_quantity": {rkaq}')
        #value = self.get(key)
        #if value is None:
        #    raise KeyError(f'No key "{key}" found')
        #    return None
        unit = self.get_fits_key_unit(key)
        unit = unit or u.dimensionless_unscaled
        return value*unit

    def set_fits_key_quantity(self, key, quantity_comment):
        """Sets `~astropy.units.Quantity` of card"""
        if isinstance(quantity_comment, tuple):
            quantity = quantity_comment[0]
            comment = quantity_comment[1]
        else:
            quantity = quantity_comment
            comment = None
        if isinstance(quantity, u.Quantity):
            value = quantity.value
            unit = quantity.unit
        else:
            value = quantity
            unit = None
        self.set(key, value, comment)
        if unit is not None:
            set_fits_key_unit(key, unit)

    #@property
    #def return_key_as_quantity(self):
    #    return self._return_key_as_quantity
    #
    #@return_key_as_quantity.setter
    #def return_key_as_quantity(self, value):
    #    value = value.lower()
    #    if value not in ['never', 'recognized', 'always']:
    #        raise ValueError('Unrecognized value for "return_key_as_quantity": {value}')
    #    self._return_key_as_quantity = value
        
    # --> These were a little too complex to surgically override
    def __getitem__(self, key):
        if isinstance(key, slice):
            return self.__class__([copy.copy(c) for c in self._cards[key]])
        elif self._haswildcard(key):
            return self.__class__([copy.copy(self._cards[idx])
                                   for idx in self._wildcardmatch(key)])
        elif isinstance(key, str):
            key = key.strip()
            if key.upper() in Card._commentary_keywords:
                key = key.upper()
                # Special case for commentary cards
                return _HeaderCommentaryCards(self, key)

        if isinstance(key, tuple):
            keyword = key[0]
        else:
            keyword = key

        card = self._cards[self._cardindex(key)]

        if card.field_specifier is not None and keyword == card.rawkeyword:
            # This is RVKC; if only the top-level keyword was specified return
            # the raw value, not the parsed out float value
            return card.rawvalue

        value = card.value
        if value == UNDEFINED:
            return None

        ############### --> New stuff <-- #############
        rkaq = self.return_key_as_quantity
        if rkaq not in ['never', 'recognized', 'always']:
            raise ValueError(f'Unrecognized value for "return_key_as_quantity": {rkaq}')
        if rkaq == 'never':
            return value
        unit = self.get_fits_key_unit(key)
        if rkaq == 'always' and unit is None:
            return value*u.dimensionless_unscaled
        if unit is None:
            return value
        return value*unit

    def __setitem__(self, key, value):
        if self._set_slice(key, value, self):
            return

        if isinstance(value, tuple):
            if len(value) > 2:
                raise ValueError(
                    'A Header item may be set with either a scalar value, '
                    'a 1-tuple containing a scalar value, or a 2-tuple '
                    'containing a scalar value and comment string.')
            if len(value) == 1:
                value, comment = value[0], None
                if value is None:
                    value = UNDEFINED
            elif len(value) == 2:
                value, comment = value
                if value is None:
                    value = UNDEFINED
                if comment is None:
                    comment = ''
        else:
            comment = None

        ############### --> New stuff <-- #############
        if isinstance(value, u.Quantity):
            unit = value.unit
            value = value.value
        else:
            unit = None

        ############### --> Original stuff <-- #############
        card = None
        if isinstance(key, int):
            card = self._cards[key]
        elif isinstance(key, tuple):
            card = self._cards[self._cardindex(key)]
        if value is None:
            value = UNDEFINED
        if card:
            card.value = value
            if comment is not None:
                card.comment = comment
            # --> insert unit somehow
            if card._modified:
                self._modified = True
        else:
            # If we get an IndexError that should be raised; we don't allow
            # assignment to non-existing indices
            self._update((key, value, comment))


####### PR2 ########
def fits_key_arithmetic(meta, operand1, operation, operand2,
                        keylist=None, handle_image=None):
    """Apply arithmetic to FITS keywords

    meta : ordered_dict

        FITS header of operand1 *after* processing by other arithmetic
        operations.  Sensible use of this feature requires
        ``handle_meta`` to be set to 'first_found' or callable that
        returns a FITS header  

    operand1 : `NDData`-like instance
        Generally the self of the calling object

    operation : callable
            The operation that is performed on the `NDData`. Supported are
            `numpy.add`, `numpy.subtract`, `numpy.multiply` and
            `numpy.true_divide`.

    operand2 : `NDData`-like instance
        Generally the self of the calling object

    keylist : list

        List of FITS keywords to apply ``operation`` to.  Each keyword
        value stands in the place of ``operand1`` and a new keyword
        value is calculated using the ``operation`` and ``operand2.``
        If ``operand2`` is an image, ``handle_image`` will be called
        to convert it to a scalar or ``None`` (see ``handle_image``)

    handle_image : callable

        Called with arguments of fits_key_arithmetic (minus
        ``handle_image``) when ``operand2`` is an image.  Return value
        of ``None`` signifies application of ``operation`` would
        nullify keywords in ``keylist,`` which are then removed.  If
        transformation of ``operand2`` into a scalar is possible

    """
    if meta is None or keylist is None:
        return meta
    # Get a list of non-None values for our keylist
    kvlist = [kv for kv in [(k, meta.get(k)) for k in keylist]
              if kv[1] is not None]
    if kvlist is None:
        return meta
    dimso2 = sum(list(operand2.shape))
    if dimso2 == 0:
        # Scalar
        o2 = operand2.data
    else:
        if handle_image is None:
            o2 = None
        else:
            o2 = handle_image(meta, operand1, operation, operand2,
                              keylist=keylist)
    for k, v in kvlist:
        if o2 is None:
            del meta[k]
            log.debug(f'Cannot express operand2 as single number, deleting {k}')
        else:
            # This code assume that units of key are the same as units
            # of operand1, so it is a good fallback if key has no
            # get_fits_key_unit
            kcomment = meta.comments[k]
            # Strip off old units, assuming they are separated by space
            kcomment, _ = kcomment.rsplit(maxsplit=1)

            # Do the calculation with or without units
            if operand1.unit is None and operand2.unit is None:
                v = operation(v, o2)
            elif operand1.unit is None:
                v = operation(v * u.dimensionless_unscaled,
                              o2 * operand2.unit)
            elif operand2.unit is None:
                v = operation(v * operand1.unit,
                              o2 * u.dimensionless_unscaled)
            else:
                v = operation(v * operand1.unit,
                              o2 * operand2.unit)

            kcomment = f'{kcomment} ({v.unit.to_string()})'
            meta[k] = (v.value, kcomment)
    return meta        

class FitsKeyArithmeticMixin(NDArithmeticMixin):
    """Mixin that adds FITS keyword arithmetic capability to `NDArithmeticMixin`

    As with the `NDArithmeticMixin`, add this after `CCDData` in the
    inheritance chain

    arithmetic_keylist : list

        List of FITS card keywords.  Arithmetic will be performed on
        values of these cards  

    """

    arithmetic_keylist = None
    handle_image = None

    def _arithmetic(self, operation, operand, **kwds):
        # Run our super to get everything it does
        result, kwargs = super()._arithmetic(operation, operand, **kwds)

        meta = kwargs['meta']
        # This essentially forces a handle_meta='first_found'.  There
        # may be a better way to fold this into _arithmetic
        if meta is None:
            meta = self.meta
        newmeta = fits_key_arithmetic(meta, self, operation, operand,
                                      keylist=self.arithmetic_keylist,
                                      handle_image=self.handle_image)
        kwargs['meta'] = newmeta
        return result, kwargs


from astropy.nddata import CCDData

print('####### PR1 ########')

# Make a basic CCD with hdr
meta = {'EXPTIME': 10,
        'CAMERA': 'SX694'}
hdr = FitsKeyQuantityHeader(meta)
ccd = CCDData(0, unit=u.dimensionless_unscaled, meta=hdr)
# Add some comment text
ccd.meta.comments['EXPTIME'] = 'Exposure time'
kcomment = ccd.meta.comments['EXPTIME']
print(f'original EXPTIME comment: "{kcomment}"')

print(ccd.meta.get_fits_key_unit('EXPTIME'))
ccd.meta.set_fits_key_unit('EXPTIME', u.s)
print(ccd.meta.get_fits_key_unit('EXPTIME'))
kcomment = ccd.meta.comments['EXPTIME']
print(kcomment)
print(ccd.meta.get_fits_key_unit('CAMERA'))

print(f'EXPTIME: {ccd.meta["EXPTIME"]}')
ccd.meta.return_key_as_quantity = 'recognized'
print(f'EXPTIME: {ccd.meta["EXPTIME"]}')

#ccd.meta['EXPTIME'] = 11

#ccd.meta['EXPTIME'] = 11*u.m
#print(f'EXPTIME: {ccd.meta["EXPTIME"]}')
ccd.meta.del_fits_key_unit('EXPTIME')
ccd.meta[0] = 11*u.m
print(f'EXPTIME: {ccd.meta["EXPTIME"]}')

#flat_fname = '/data/io/IoIO/reduced/Calibration/2020-03-22_B_flat.fits'
#
#ccd = CCDData.read(flat_fname)
#class Test(FitsKeyQuantityMixin, CCDData):
#    pass
#ccd = Test.read(flat_fname)
#unit = ccd.get_fits_key_unit('EXPTIME')
#print(unit)
#unit = ccd.get_fits_key_unit('GAIN')
#print(unit)
#unit = ccd.get_fits_key_unit('SATLEVEL')
#print(unit)
#ccd.set_fits_key_unit('SATLEVEL', u.m)
#print(ccd.meta.comments['SATLEVEL'])
#
#ccd.del_fits_key_unit('SATLEVEL')
#print(ccd.meta.comments['SATLEVEL'])
#
#ccd.set_fits_key_unit('SATLEVEL', unit)
#print(ccd.meta.comments['SATLEVEL'])
#
#print(ccd.get_fits_key_quantity('SATLEVEL'))
#
#kcomment = ccd.meta.comments['OVERSCAN_VALUE']
#print(kcomment)
#ccd.meta.comments['OVERSCAN_VALUE'] = 'make this long' + kcomment 
#
#kcomment = ccd.meta.comments['OVERSCAN_VALUE']
#print(kcomment)
#
#unit = u.electron
#ccd.set_fits_key_unit('OVERSCAN_VALUE', unit)
#print(ccd.meta.comments['OVERSCAN_VALUE'])


#set_fits_key_quantity(key, quantity_comment, meta)



#class Test(FitsKeyArithmeticMixin, CCDData):
#    def __init__(self, *args, **kwargs):
#        super().__init__(*args, **kwargs)
#        self.arithmetic_keylist = ['satlevel', 'nonlin']
##ccd = Test.read(flat_fname)
###print(ccd.meta)
###ccdd = ccd.divide(10, handle_meta='first_found')
###ccdd = ccd.divide(10*u.dimensionless_unscaled, handle_meta='first_found')
###ccdd = ccd.divide(10, handle_meta='first_found')
##ccdd = ccd.divide(10)
##print(ccdd.meta)
#
###kcomment = 'this is a comment (electrons)'
###kcomment = 'this is a comment (electron)'
##kcomment = 'this is a comment'
### Find () at very end of line which is our format for a unit designation
##m = re.search(r'(.)$', kcomment)
##m = re.search(r"\((\w+)\)", kcomment)
##if m is None:
##    log.debug(f'no unit in {kcomment}')
##punit_str = m.group(0)
##unit_str = re.sub('[()]', '', punit_str)
##try:
##    unit = u.Unit(unit_str)
##except ValueError as e:
##    log.warning(e)
##    unit = None
##print(unit_str)
#
#flat_fname = '/data/io/IoIO/reduced/Calibration/2020-03-22_B_flat.fits'
#ccd = CCDData.read(flat_fname)
#unit = get_fits_key_unit('EXPTIME', ccd.meta)
#unit = get_fits_key_unit('GAIN', ccd.meta)
#print(ccd.meta.comments['SATLEVEL'])
#unit = get_fits_key_unit('SATLEVEL', ccd.meta)
#print(unit)
#
#set_fits_key_unit('SATLEVEL', u.m, ccd.meta)
#print(ccd.meta.comments['SATLEVEL'])
#
#
#del_fits_key_unit('SATLEVEL', ccd.meta)
#print(ccd.meta.comments['SATLEVEL'])
#
#set_fits_key_unit('SATLEVEL', unit, ccd.meta)
#print(ccd.meta.comments['SATLEVEL'])
#
#print(get_fits_key_quantity('SATLEVEL', ccd.meta))
#
#key = 'SATLEVEL'
#value = 19081.35378864727
#kcomment = ccd.meta.comments[key]
##c = fits.Card(key, value, kcomment+kcomment)
#c = fits.Card(key, value, kcomment)
#
#kcomment = ccd.meta.comments['OVERSCAN_VALUE']
#print(kcomment)
#ccd.meta.comments['OVERSCAN_VALUE'] = 'make this long' + kcomment 
#
#kcomment = ccd.meta.comments['OVERSCAN_VALUE']
#print(kcomment)
#
#
#unit = u.electron
#set_fits_key_unit('OVERSCAN_VALUE', unit, ccd.meta)
#print(ccd.meta.comments['OVERSCAN_VALUE'])
#
##set_fits_key_quantity(key, quantity_comment, meta)
#
