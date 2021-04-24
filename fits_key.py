"""Supplements to astropy.nddata
"""
import re
import warnings

import astropy.units as u
from astropy import log
from astropy.io import fits
from astropy.nddata import NDArithmeticMixin

# Create the unit regular expression string from configurable pieces.
# Default is r'\ \(.*\)$' which matches " (<unit>)" at the end of the FITS
# card comment
UNIT_STR_START = ' '
UNIT_STR_DELIMETERS = '()'
UNIT_STR_END = ''
UNIT_STR_POSITION = 'end'
s = re.escape(UNIT_STR_START)
l = re.escape(UNIT_STR_DELIMETERS[0])
r = re.escape(UNIT_STR_DELIMETERS[1])
e = re.escape(UNIT_STR_END)
UNIT_STR_POSITION = UNIT_STR_POSITION.lower()
if UNIT_STR_POSITION in ['start', 'beginning']:
    ms = '^'
    me = ''
if UNIT_STR_POSITION == 'end':
    ms = ''
    me = '$'
KCOMMENT_UNIT_REGEXP = f'{ms}{s}{l}.*{r}{e}{me}'



def get_fits_key_unit(key, meta):
    if meta.get(key) is None:
        raise KeyError(f'No key "{key}" found')
    kcomment = meta.comments[key]
    # Extract unit together with lead-in and delimiter
    # https://stackoverflow.com/questions/8569201/get-the-string-within-brackets-in-python
    m = re.search(KCOMMENT_UNIT_REGEXP, kcomment)
    if m is None:
        log.debug(f'no unit matching "(unit)" in "{kcomment}"')
        return None
    # Strip off delemeters
    punit_str = m.group(0)
    # No escaping needed because within re '[]'
    s = UNIT_STR_START
    l = UNIT_STR_DELIMETERS[0]
    r = UNIT_STR_DELIMETERS[1]
    e = UNIT_STR_END
    unit_str = re.sub(f'[{s}{l}{r}{e}]', '', punit_str)
    try:
        unit = u.Unit(unit_str)
    except ValueError as e:
        log.warning(kcomment)
        log.warning(e)
        return None
    return unit

def del_fits_key_unit(key, meta):
    if get_fits_key_unit(key, meta) is None:
        raise ValueError(f'No unit for key "{key}" to delete')
    kcomment = meta.comments[key]
    kcomment = re.sub(KCOMMENT_UNIT_REGEXP, '', kcomment)
    meta.comments[key] = kcomment
    return True

def set_fits_key_unit(key, unit, meta):
    value = meta.get(key)
    if value is None:
        raise KeyError('No key "{key}" found')
    if not isinstance(unit, u.UnitBase):
        raise ValueError('unit is not an instance of astropy.units.Unit')
    try:
        del_fits_key_unit(key, meta)
    except ValueError:
        pass
    kcomment = meta.comments[key]
    unit_str = unit.to_string()
    # Calculate how much room we need for the unit on the end of
    # kcomment so we can truncate the comment if necessary.  Doing it
    # this way lets astropy handle the HEIRARCH stuff, which moves the
    # start column of the comment
    uroom = len(unit_str) + 1    
    # Use raw fits.Card object to calculate the card image length to
    # make sure our unit will fit in.  Ignore warning about converting
    # cards to HEIRARCH
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=fits.verify.VerifyWarning)
        c = fits.Card(key, value, kcomment)
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
    meta.comments[key] = kcomment

def get_fits_key_quantity(key, meta):
    value = meta.get(key)
    if value is None:
        log.warning(f'No key "{key}" found')
        return None
    unit = get_fits_key_unit(key, meta)
    unit = unit or u.dimensionless_unscaled
    return value*unit

def set_fits_key_quantity(key, quantity_comment, meta):
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
    meta.set(key, value, comment)
    if unit is not None:
        set_fits_key_unit(key, unit)

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

flat_fname = '/data/io/IoIO/reduced/Calibration/2020-03-22_B_flat.fits'
class Test(FitsKeyArithmeticMixin, CCDData):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.arithmetic_keylist = ['satlevel', 'nonlin']

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
flat_fname = '/data/io/IoIO/reduced/Calibration/2020-03-22_B_flat.fits'
ccd = CCDData.read(flat_fname)
unit = get_fits_key_unit('EXPTIME', ccd.meta)
unit = get_fits_key_unit('GAIN', ccd.meta)
print(ccd.meta.comments['SATLEVEL'])
unit = get_fits_key_unit('SATLEVEL', ccd.meta)
print(unit)

set_fits_key_unit('SATLEVEL', u.m, ccd.meta)
print(ccd.meta.comments['SATLEVEL'])


del_fits_key_unit('SATLEVEL', ccd.meta)
print(ccd.meta.comments['SATLEVEL'])

set_fits_key_unit('SATLEVEL', unit, ccd.meta)
print(ccd.meta.comments['SATLEVEL'])

print(get_fits_key_quantity('SATLEVEL', ccd.meta))

key = 'SATLEVEL'
value = 19081.35378864727
kcomment = ccd.meta.comments[key]
#c = fits.Card(key, value, kcomment+kcomment)
c = fits.Card(key, value, kcomment)

kcomment = ccd.meta.comments['OVERSCAN_VALUE']
print(kcomment)
ccd.meta.comments['OVERSCAN_VALUE'] = 'make this long' + kcomment 

kcomment = ccd.meta.comments['OVERSCAN_VALUE']
print(kcomment)


unit = u.electron
set_fits_key_unit('OVERSCAN_VALUE', unit, ccd.meta)
print(ccd.meta.comments['OVERSCAN_VALUE'])

#set_fits_key_quantity(key, quantity_comment, meta)
