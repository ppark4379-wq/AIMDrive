import math
from .Common import _Receiver, _Packet

GGA_FIX_QUALITIES = ('1', '2', '4', '5')

class GpsReceiver(_Receiver):
    """NMEA0183 GGA/RMC. Only checksum-valid, in-range fixes update the position."""

    def __init__(self, port, **kwargs):
        self._fix = {"latitude": None, "longitude": None, "altitude": None}
        super().__init__(port, recv_bytes=4096, **kwargs)

    @staticmethod
    def _degrees(value, hemisphere, allowed):
        if not value:
            return None
        if hemisphere not in allowed:
            raise ValueError('Invalid NMEA hemisphere')
        dot = value.find(".")
        if dot < 3:
            raise ValueError('Invalid NMEA coordinate')
        degrees = float(value[: dot - 2])
        minutes = float(value[dot - 2:])
        result = degrees + minutes / 60.0
        return -result if hemisphere in ("S", "W") else result

    @staticmethod
    def _verified_body(sentence):
        """Sentence without '*hh'; None when a present checksum does not match."""
        if '*' not in sentence:
            return sentence
        body, checksum = sentence.split('*', 1)
        computed = 0
        for char in body:
            computed ^= ord(char)
        try:
            return body if computed == int(checksum[:2], 16) else None
        except ValueError:
            return None

    def _parse(self, parts):
        """(latitude, longitude, altitude-or-None) of a usable fix, else None."""
        if parts[0].endswith("GGA") and len(parts) > 9:
            if parts[6] not in GGA_FIX_QUALITIES:
                return None
            latitude = self._degrees(parts[2], parts[3], ('N', 'S'))
            longitude = self._degrees(parts[4], parts[5], ('E', 'W'))
            try:
                altitude = float(parts[9])
            except ValueError:
                altitude = None
            return latitude, longitude, altitude
        if parts[0].endswith("RMC") and len(parts) > 6:
            if parts[2] != 'A':
                return None
            return (self._degrees(parts[3], parts[4], ('N', 'S')),
                    self._degrees(parts[5], parts[6], ('E', 'W')), None)
        return None

    @staticmethod
    def _valid_coordinates(latitude, longitude):
        return (latitude is not None and longitude is not None
                and math.isfinite(latitude) and math.isfinite(longitude)
                and -90 <= latitude <= 90 and -180 <= longitude <= 180
                and not (latitude == 0 and longitude == 0))

    def _consume(self, payload):
        text = payload.decode("ascii", "ignore")
        updated = False
        for sentence in text.split("$"):
            sentence = sentence.strip()
            if not sentence:
                continue
            body = self._verified_body(sentence)
            if body is None:
                self._count('checksum_mismatches')
                continue
            parts = body.split(",")
            if len(parts) < 6:
                continue
            try:
                fix = self._parse(parts)
            except ValueError:
                self._count('malformed_sentences')
                continue
            if fix is None:
                continue
            latitude, longitude, altitude = fix
            if not self._valid_coordinates(latitude, longitude):
                self._count('invalid_coordinates')
                continue
            self._fix.update(latitude=latitude, longitude=longitude)
            if altitude is not None and math.isfinite(altitude):
                self._fix["altitude"] = altitude
            updated = True
        if updated and self._fix["latitude"] is not None:
            return dict(self._fix)
        return None
