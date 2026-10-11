from .Common import _Receiver, _Packet

class GpsReceiver(_Receiver):
    def __init__(self, port, **kwargs):
        self._fix = {"latitude": None, "longitude": None, "altitude": None}
        super().__init__(port, recv_bytes=4096, **kwargs)

    @staticmethod
    def _degrees(value, hemisphere):
        if not value:
            return None
        dot = value.find(".")
        degrees = float(value[: dot - 2])
        minutes = float(value[dot - 2:])
        result = degrees + minutes / 60.0
        return -result if hemisphere in ("S", "W") else result

    def _consume(self, payload):
        text = payload.decode("ascii", "ignore")
        updated = False
        for sentence in text.split("$"):
            parts = sentence.strip().split(",")
            if len(parts) < 6:
                continue
            if parts[0].endswith("GGA") and len(parts) > 9:
                if len(parts) < 7 or parts[6] not in ('1', '2', '4', '5'):
                    continue
                latitude = self._degrees(parts[2], parts[3])
                longitude = self._degrees(parts[4], parts[5])
                if latitude is not None and longitude is not None:
                    self._fix.update(latitude=latitude, longitude=longitude)
                    try:
                        self._fix["altitude"] = float(parts[9])
                    except ValueError:
                        pass
                    updated = True
            elif parts[0].endswith("RMC") and len(parts) > 6:
                if parts[2] != 'A':
                    continue
                latitude = self._degrees(parts[3], parts[4])
                longitude = self._degrees(parts[5], parts[6])
                if latitude is not None and longitude is not None:
                    self._fix.update(latitude=latitude, longitude=longitude)
                    updated = True
        if updated and self._fix["latitude"] is not None:
            return dict(self._fix)
        return None
