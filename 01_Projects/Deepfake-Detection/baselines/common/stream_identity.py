"""Conservative compressed-stream identity for unchanged-modality feature reuse.

Packet bytes, codec configuration, time origin and rational timestamps must all
match. A remux that fails this test simply runs normal preparation/inference.
"""
import hashlib,json
from fractions import Fraction


def stream_identity(path,kind):
    import av
    if kind not in ('audio','video'):raise ValueError('Expected audio or video')
    with av.open(str(path)) as container:
        streams=[s for s in container.streams if s.type==kind]
        if len(streams)!=1:return None
        stream=streams[0];codec=stream.codec_context
        origin=Fraction(container.start_time or 0,av.time_base)
        def timestamp(value,time_base):
            if value is None:return None
            value=Fraction(value)*Fraction(time_base)-origin
            return [value.numerator,value.denominator]
        header={'kind':kind,'codec':codec.name,'extradata':hashlib.sha256(codec.extradata or b'').hexdigest(),
            'start':timestamp(stream.start_time,stream.time_base),'origin':[origin.numerator,origin.denominator],
            'sample_rate':getattr(codec,'sample_rate',None),'width':getattr(codec,'width',None),
            'height':getattr(codec,'height',None),'layout':str(getattr(codec,'layout',None)),
            'format':str(getattr(codec,'format',None))}
        digest=hashlib.sha256(json.dumps(header,sort_keys=True).encode());count=0
        for packet in container.demux(stream):
            if not packet.size:continue
            payload=bytes(packet)
            stamp={'pts':timestamp(packet.pts,packet.time_base),'dts':timestamp(packet.dts,packet.time_base),
                'duration':None if packet.duration is None else str(Fraction(packet.duration)*Fraction(packet.time_base)),
                'size':len(payload),'keyframe':packet.is_keyframe}
            digest.update(json.dumps(stamp,sort_keys=True).encode());digest.update(payload);count+=1
        return {'sha256':digest.hexdigest(),'packets':count,'kind':kind} if count else None
