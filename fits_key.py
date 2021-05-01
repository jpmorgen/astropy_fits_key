"""Supplements to astropy.nddata
"""
import re
import warnings

import numpy as np

import astropy.units as u
from astropy.units import Quantity
from astropy import log
from astropy.io import fits
from astropy.io.fits import conf, Header
from astropy.io.fits.card import Card, UNDEFINED, VALUE_INDICATOR, Undefined
from astropy.io.fits.verify import VerifyError, VerifyWarning
from astropy.nddata import CCDData, NDArithmeticMixin

# Create the unit regular expression string from configurable pieces.
# Below implements r'\ \(.*\)$' which matches " (<unit>)" at the end
# of the FITS card comment

####### PR1 draft ########
class QuantityCard(Card):
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

        ############### --> New stuff <-- #############
        self._unit = None

        # --> Again, these would be conf items
        self.return_key_as_quantity = return_key_as_quantity or self.return_key_as_quantity
        self.unit_str_start = unit_str_start or self.unit_str_start
        self.unit_str_delimeters = unit_str_delimeters or self.unit_str_delimeters
        self.unit_str_end = unit_str_end or self.unit_str_end
        self.unit_str_position = unit_str_position or self.unit_str_position
    
        ############### --> Original stuff <-- #############

        super().__init__(*args, **kwargs)

    @property
    def value(self):
        """The value associated with the keyword stored in this card."""

        if self.field_specifier:
            return float(self._value)

        if self._value is not None:
            value = self._value
        elif self._valuestring is not None or self._image:
            value = self._value = self._parse_value()
        else:
            if self._keyword == '':
                self._value = value = ''
            else:
                self._value = value = UNDEFINED

        if conf.strip_header_whitespace and isinstance(value, str):
            value = value.rstrip()

        ############### --> New stuff <-- #############
        rkaq = self.return_key_as_quantity.lower()
        if rkaq == 'never':
            pass
        elif rkaq == 'recognized':
            if self.unit is None:
                pass
            else:
                value *= self.unit
        elif rkaq == 'always':
            if self.unit is None:
                value *= u.dimensionless_unscaled
            else:
                value *= self.unit
        else:
            raise ValueError(f'Unrecognized value for "return_key_as_quantity": {rkaq}')
        ############### --> Original stuff <-- #############

        return value

    @value.setter
    def value(self, value):
        if self._invalid:
            raise ValueError(
                'The value of invalid/unparseable cards cannot set.  Either '
                'delete this card from the header or replace it.')

        if value is None:
            value = UNDEFINED

        try:
            oldvalue = self.value
        except VerifyError:
            # probably a parsing error, falling back to the internal _value
            # which should be None. This may happen while calling _fix_value.
            oldvalue = self._value

        if oldvalue is None:
            oldvalue = UNDEFINED

        ############### --> New stuff <-- #############
        if isinstance(value, Quantity):
            self._unit = value.unit
            value = value.value
        else:
            self._unit = None
            self.comment = re.sub(self._full_unit_regexp, '', self.comment)
        ############### --> Original stuff <-- #############

        if not isinstance(value,
                          (str, int, float, complex, bool, Undefined,
                           np.floating, np.integer, np.complexfloating,
                           np.bool_)):
            raise ValueError(f'Illegal value: {value!r}.')

        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            raise ValueError("Floating point {!r} values are not allowed "
                             "in FITS headers.".format(value))

        elif isinstance(value, str):
            m = self._ascii_text_re.match(value)
            if not m:
                raise ValueError(
                    'FITS header values must contain standard printable ASCII '
                    'characters; {!r} contains characters not representable in '
                    'ASCII or non-printable characters.'.format(value))
        elif isinstance(value, np.bool_):
            value = bool(value)

        if (conf.strip_header_whitespace and
                (isinstance(oldvalue, str) and isinstance(value, str))):
            # Ignore extra whitespace when comparing the new value to the old
            different = oldvalue.rstrip() != value.rstrip()
        elif isinstance(oldvalue, bool) or isinstance(value, bool):
            different = oldvalue is not value
        else:
            different = (oldvalue != value or
                         not isinstance(value, type(oldvalue)))

        if different:
            self._value = value
            self._rawvalue = None
            self._modified = True
            self._valuestring = None
            self._valuemodified = True
            if self.field_specifier:
                try:
                    self._value = _int_or_float(self._value)
                except ValueError:
                    raise ValueError('value {} is not a float'.format(
                            self._value))

    @value.deleter
    def value(self):
        if self._invalid:
            raise ValueError(
                'The value of invalid/unparseable cards cannot deleted.  '
                'Either delete this card from the header or replace it.')

        if not self.field_specifier:
            self.value = ''
            ############### --> New stuff <-- #############
            self._unit = None
            # --> Should this be self._comment?
            self.comment = re.sub(self._full_unit_regexp, '', self.comment)
        ############### --> Original stuff <-- #############
        else:
            raise AttributeError('Values cannot be deleted from record-valued '
                                 'keyword cards')

    ############### --> Original stuff <-- #############
    @property
    def comment(self):
        """Get the comment attribute from the card image if not already set."""

        if self._comment is not None:
            return self._comment
        elif self._image:
            self._comment = self._parse_comment()
            return self._comment
        else:
            self._comment = ''
            return ''

    @comment.setter
    def comment(self, comment):
        if self._invalid:
            raise ValueError(
                'The comment of invalid/unparseable cards cannot set.  Either '
                'delete this card from the header or replace it.')

        if comment is None:
            comment = ''

        if isinstance(comment, str):
            m = self._ascii_text_re.match(comment)
            if not m:
                raise ValueError(
                    'FITS header comments must contain standard printable '
                    'ASCII characters; {!r} contains characters not '
                    'representable in ASCII or non-printable characters.'
                    .format(comment))

        try:
            oldcomment = self.comment
        except VerifyError:
            # probably a parsing error, falling back to the internal _comment
            # which should be None.
            oldcomment = self._comment

        if oldcomment is None:
            oldcomment = ''
        if comment != oldcomment:
            self._comment = comment
            ############### --> New stuff <-- #############
            oldunit = self._unit
            newunit = self._get_comment_unit(comment)
            if newunit != oldunit:
                # Change the card unit, but raise warning -- input via
                # Quantity should be prefered
                self._unit = newunit
                if oldunit:
                    oldunit_str = oldunit.to_string()
                else:
                    oldunit_str = 'None'
                if newunit:
                    newunit_str = newunit.to_string()
                else:
                    newunit_str = 'None'
                # --> Not sure why this is not working
                warnings.warn(
                    'Changing unit of card from {} to '
                    '{}'.format(
                        oldunit_str, newunit_str), VerifyWarning)
                ## Not sure if there is some new standard to use f strings
                #warnings.warn(
                #    f'Changing unit of card from {oldunit_str} to '
                #    f'{newunit_str}', VerifyWarning)
                #print(
                #    'Changing unit of card from {} to '
                #    '{}'.format(
                #        oldunit_str, newunit_str))
                print(
                    f'Changing unit of card from {oldunit_str} to '
                    f'{newunit_str}')
            ############### --> Original stuff <-- #############
            self._modified = True

    ############### --> Original stuff <-- #############
    @comment.deleter
    def comment(self):
        if self._invalid:
            raise ValueError(
                'The comment of invalid/unparseable cards cannot deleted.  '
                'Either delete this card from the header or replace it.')

        self.comment = ''
        ############### --> New stuff <-- #############
        self._unit = None
        ############### --> Original stuff <-- #############

    ############### --> New stuff <-- #############
    @property
    def unit(self):
        return self._unit

    @unit.setter
    def unit(self, unit):
        self._unit = unit
        comment = self._set_comment_unit(self.comment, unit)

    @unit.deleter
    def unit(self):
        if self._invalid:
            raise ValueError(
                'The comment of invalid/unparseable cards cannot deleted.  '
                'Either delete this card from the header or replace it.')
        self._unit = None
        self.comment = re.sub(self._full_unit_regexp, '', self.comment)

    ############### --> Original stuff <-- #############
    def _parse_comment(self):
        comment = super()._parse_comment()
        ############### --> New stuff <-- #############
        self._unit = self._get_comment_unit(comment)
        ############### --> Original stuff <-- #############
        return comment

    ############### --> New stuff <-- #############
    @property
    def _full_unit_regexp(self):
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

    def _get_comment_unit(self, comment):
        """Returns `~astropy.units.Unit` found in comment, if defined, else ``None``"""
        # Extract unit together with lead-in and delimiter
        # https://stackoverflow.com/questions/8569201/get-the-string-within-brackets-in-python
        m = re.search(self._full_unit_regexp, comment)
        if m is None:
            log.debug(f'no unit matching "{self._full_unit_regexp}" in "{comment}"')
            return None
        # Strip off delemeters
        punit_str = m.group(0)
        s = re.escape(self.unit_str_start)
        l = re.escape(self.unit_str_delimeters[0])
        r = re.escape(self.unit_str_delimeters[1])
        e = re.escape(self.unit_str_end)
        print(f'[{s}{l}{r}{e}]', '', punit_str)
        unit_str = re.sub(f'[{s}{l}{r}{e}]', '', punit_str)
        try:
            unit = u.Unit(unit_str)
        except ValueError as e:
            log.warning(f'Unit conversion error in comment {comment}: {e}')
            return None
        return unit

    def _full_unit_str(self, unit):
        unit_str = unit.to_string()
        s = self.unit_str_start
        l = self.unit_str_delimeters[0]
        r = self.unit_str_delimeters[1]
        e = self.unit_str_end
        return f'{s}{l}{unit_str}{r}{e}'

    def _set_comment_unit(self, comment, unit):
        """put unit str into comment"""
        if unit is None:
            del self.unit
            return self.comment
        fus = self._full_unit_str(unit)
        p = self.unit_str_position 
        if p.lower() in ['start', 'beginning']:
            comment = f'{fus}{comment}'
        elif p.lower() in ['end']:
            comment = f'{comment}{fus}'
        else:
            raise ValueError(f'unrecognized unit string position {p}')
        return comment

    ############### --> New stuff & modifications <-- #############
    def _format_image(self, card_length=None):
        ############### --> New stuff <-- #############
        if card_length is None and self._unit:
            # Clear off unit from comment property, which might extend
            # beyond the end of the legal card, create a card image
            # with enough room for the unit string and add the unit
            # string back into the comment.  Doing this clears
            # self._unit, so save it
            unit = self._unit
            fus = self._full_unit_str(unit)
            uroom = len(fus)
            card_length = self.length - uroom
            self.comment = re.sub(self._full_unit_regexp, '', self.comment)
            # Save image to avoid extra calls in _split
            self._image = self._format_image(card_length)
            card_length = self.length
            # Now we want to add the unit directly to the comment.
            comment = self._parse_comment()
            self.comment = self._set_comment_unit(comment, unit)
        elif card_length is None:
            card_length = self.length

        ############### --> New stuff <-- #############
            
        ############### --> modifications for card_length <-- #############
        keyword = self._format_keyword()

        value = self._format_value()
        is_commentary = keyword.strip() in self._commentary_keywords
        if is_commentary:
            comment = ''
        else:
            comment = self._format_comment()

        # equal sign string
        # by default use the standard value indicator even for HIERARCH cards;
        # later we may abbreviate it if necessary
        delimiter = VALUE_INDICATOR
        if is_commentary:
            delimiter = ''

        # put all parts together
        output = ''.join([keyword, delimiter, value, comment])

        # For HIERARCH cards we can save a bit of space if necessary by
        # removing the space between the keyword and the equals sign; I'm
        # guessing this is part of the HIEARCH card specification
        keywordvalue_length = len(keyword) + len(delimiter) + len(value)
        if (keywordvalue_length > card_length and
                keyword.startswith('HIERARCH')):
            if (keywordvalue_length == card_length + 1 and keyword[-1] == ' '):
                output = ''.join([keyword[:-1], delimiter, value, comment])
            else:
                # I guess the HIERARCH card spec is incompatible with CONTINUE
                # cards
                raise ValueError('The header keyword {!r} with its value is '
                                 'too long'.format(self.keyword))

        if len(output) <= card_length:
            output = ('{0:' + str(card_length) + '}').format(output)
        else:
            # longstring case (CONTINUE card)
            # try not to use CONTINUE if the string value can fit in one line.
            # Instead, just truncate the comment
            if (isinstance(self.value, str) and
                    len(value) > (card_length - 10)):
                output = self._format_long_image()
            else:
                warnings.warn('Card is too long, comment will be truncated.',
                              VerifyWarning)
                output = output[:card_length]
        return output
        ############### --> modifications for card_length <-- #############


print('####### PR1 ########')

#tc = Card('EXPTIME', 10, 'Exposure time in seconds')
#print("Card('EXPTIME', 10, 'Exposure time in seconds')")
#print(tc.image)
#print(tc.value)
#print(tc.comment)

print('+++Create plain card')
print(">>> c = QuantityCard('EXPTIME', 10, 'Exposure time in seconds')")
c = QuantityCard('EXPTIME', 10, 'Exposure time in seconds')
print(c.image)
print(c.value)
print(c.comment)

print('+++Create card with units in comment')
print(">>> c = QuantityCard('EXPTIME', 10, 'Exposure time (s)')")
c = QuantityCard('EXPTIME', 10, 'Exposure time (s)')

#rkaq = 'always'
rkaq = 'recognized'
c.return_key_as_quantity = rkaq

print('print(c.image)')
print(c.image)
print('print(c.value)')
print(c.value)
print('print(c.comment)')
print(c.comment)


print('+++Change to non-unitful value')
print('>>> c.value = 20')
c.value = 20
print('print(c.image)')
print(c.image)
print('print(c.value)')
print(c.value)
print('print(c.comment)')
print(c.comment)

print('+++Change to value as Quantity')
print('>>> c.value = 11*u.s')
c.value = 11*u.s
print('print(c.image)')
print(c.image)
print('print(c.value)')
print(c.value)
print('print(c.comment)')
print(c.comment)

print('+++Change the comment, nuking the units')
print(">>> c.comment = 'This is the start of a really long comment'")
c.comment = 'This is the start of a really long comment'
print('print(c.image)')
print(c.image)
print('print(c.value)')
print(c.value)
print('print(c.comment)')
print(c.comment)

print('+++Put units back in with new unit property')
print('>>> c.unit = u.s')
c.unit = u.s
print('print(c.image)')
print(c.image)
print('print(c.value)')
print(c.value)
print('print(c.comment)')
print(c.comment)

print('+++Extend comment')
print(">>> c.comment = 'This is the creation of a reallyreallyreallylongcomment'")
c.comment = 'This is the creation of a reallyreallyreallylongcomment'
print('print(c.image)')
print(c.image)
print('print(c.value)')
print(c.value)
print('print(c.comment)')
print(c.comment)

print('+++Put units back into extended comment')
print('>>> c.unit = u.s')
c.unit = u.s
print('print(c.image)')
print(c.image)
print('print(c.value)')
print(c.value)
print('print(c.comment)')
print(c.comment)

print('+++Set units to None')
print('>>> c.unit = None')
c.unit = None
print('print(c.image)')
print(c.image)
print('print(c.value)')
print(c.value)
print('print(c.comment)')
print(c.comment)

print('+++Change unit location to beginning of comment')
c.return_key_as_quantity = 'always'
c.unit_str_start = ''
c.unit_str_delimeters = '[]'
c.unit_str_end = ' '
c.unit_str_position = 'start'

print('+++Put new unit position back into extended comment')
print('>>> c.unit = u.s')
c.unit = u.s
print('print(c.image)')
print(c.image)
print('print(c.value)')
print(c.value)
print('print(c.comment)')
print(c.comment)

print('+++Put into header')
print('hdr = Header([c])')
hdr = Header([c])
print("hdr['EXPTIME']")
print(hdr['EXPTIME'])
print("hdr.comments['EXPTIME']")
print(hdr.comments['EXPTIME'])
print("hdr.cards['EXPTIME'].unit")
print(hdr.cards['EXPTIME'].unit)

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
            comment = meta.comments[k]
            # Strip off old units, assuming they are separated by space
            comment, _ = comment.rsplit(maxsplit=1)

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

            comment = f'{comment} ({v.unit.to_string()})'
            meta[k] = (v.value, comment)
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




## # Make a basic CCD with hdr
## meta = {'EXPTIME': 10,
##         'CAMERA': 'SX694'}
## hdr = FitsKeyQuantityHeader(meta)
## ccd = CCDData(0, unit=u.dimensionless_unscaled, meta=hdr)
## # Add some comment text
## ccd.meta.comments['EXPTIME'] = 'Exposure time'
## comment = ccd.meta.comments['EXPTIME']
## print(f'original EXPTIME comment: "{comment}"')
## 
## print(ccd.meta.get_fits_key_unit('EXPTIME'))
## ccd.meta.set_fits_key_unit('EXPTIME', u.s)
## print(ccd.meta.get_fits_key_unit('EXPTIME'))
## comment = ccd.meta.comments['EXPTIME']
## print(comment)
## print(ccd.meta.get_fits_key_unit('CAMERA'))
## 
## print(f'EXPTIME: {ccd.meta["EXPTIME"]}')
## ccd.meta.return_key_as_quantity = 'recognized'
## print(f'EXPTIME: {ccd.meta["EXPTIME"]}')
## 
## #ccd.meta['EXPTIME'] = 11
## 
## #ccd.meta['EXPTIME'] = 11*u.m
## #print(f'EXPTIME: {ccd.meta["EXPTIME"]}')
## ccd.meta.del_fits_key_unit('EXPTIME')
## ccd.meta[0] = 11*u.m
## print(f'EXPTIME: {ccd.meta["EXPTIME"]}')
##


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
#comment = ccd.meta.comments['OVERSCAN_VALUE']
#print(comment)
#ccd.meta.comments['OVERSCAN_VALUE'] = 'make this long' + comment 
#
#comment = ccd.meta.comments['OVERSCAN_VALUE']
#print(comment)
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
###comment = 'this is a comment (electrons)'
###comment = 'this is a comment (electron)'
##comment = 'this is a comment'
### Find () at very end of line which is our format for a unit designation
##m = re.search(r'(.)$', comment)
##m = re.search(r"\((\w+)\)", comment)
##if m is None:
##    log.debug(f'no unit in {comment}')
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
#comment = ccd.meta.comments[key]
##c = fits.Card(key, value, comment+comment)
#c = fits.Card(key, value, comment)
#
#comment = ccd.meta.comments['OVERSCAN_VALUE']
#print(comment)
#ccd.meta.comments['OVERSCAN_VALUE'] = 'make this long' + comment 
#
#comment = ccd.meta.comments['OVERSCAN_VALUE']
#print(comment)
#
#
#unit = u.electron
#set_fits_key_unit('OVERSCAN_VALUE', unit, ccd.meta)
#print(ccd.meta.comments['OVERSCAN_VALUE'])
#
##set_fits_key_quantity(key, quantity_comment, meta)
#

