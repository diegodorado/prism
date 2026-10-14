use rosc::{OscMessage, OscPacket, OscType};

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum OscIn {
    Active(i32),
    Fps(f32),
    OutputName(String),
    InputName(String),
    Frames(i32),
    Error { address: String, reason: String },
    Unknown,
}

pub fn decode(packet: &OscPacket) -> Option<OscIn> {
    match packet {
        OscPacket::Message(msg) => decode_msg(msg),
        OscPacket::Bundle(_) => None,
    }
}

fn decode_msg(msg: &OscMessage) -> Option<OscIn> {
    let addr = msg.addr.as_str();
    match addr {
        "/active" => get_i32(&msg.args, 0).map(OscIn::Active),
        "/fps" => get_f32(&msg.args, 0).map(OscIn::Fps),
        "/output_name" => get_string(&msg.args, 0).map(OscIn::OutputName),
        "/input_name" => get_string(&msg.args, 0).map(OscIn::InputName),
        "/frames" => get_i32(&msg.args, 0).map(OscIn::Frames),
        "/error" => {
            if msg.args.len() >= 2 {
                if let (Some(a), Some(r)) = (get_string(&msg.args, 0), get_string(&msg.args, 1)) {
                    return Some(OscIn::Error {
                        address: a,
                        reason: r,
                    });
                }
            }
            None
        }
        _ => Some(OscIn::Unknown),
    }
}

pub fn encode_prompt(prompt: &str) -> OscMessage {
    OscMessage {
        addr: "/prompt".into(),
        args: vec![OscType::String(prompt.to_string())],
    }
}

pub fn encode_negative(neg: &str) -> OscMessage {
    OscMessage {
        addr: "/negative".into(),
        args: vec![OscType::String(neg.to_string())],
    }
}

pub fn encode_seed(seed: i32) -> OscMessage {
    OscMessage {
        addr: "/seed".into(),
        args: vec![OscType::Int(seed)],
    }
}

pub fn encode_guidance(g: f32) -> OscMessage {
    OscMessage {
        addr: "/guidance".into(),
        args: vec![OscType::Float(g)],
    }
}

pub fn encode_delta(d: f32) -> OscMessage {
    OscMessage {
        addr: "/delta".into(),
        args: vec![OscType::Float(d)],
    }
}

pub fn encode_steps(steps: &[i32]) -> OscMessage {
    OscMessage {
        addr: "/steps".into(),
        args: steps.iter().map(|v| OscType::Int(*v)).collect(),
    }
}

pub fn encode_maxfps(m: i32) -> OscMessage {
    OscMessage {
        addr: "/maxfps".into(),
        args: vec![OscType::Int(m)],
    }
}

pub fn encode_input(s: &str) -> OscMessage {
    OscMessage {
        addr: "/input".into(),
        args: vec![OscType::String(s.to_string())],
    }
}

pub fn encode_play() -> OscMessage {
    OscMessage {
        addr: "/play".into(),
        args: vec![],
    }
}

pub fn encode_pause() -> OscMessage {
    OscMessage {
        addr: "/pause".into(),
        args: vec![],
    }
}

pub fn encode_stop() -> OscMessage {
    OscMessage {
        addr: "/stop".into(),
        args: vec![],
    }
}

pub fn encode_unload() -> OscMessage {
    OscMessage {
        addr: "/unload".into(),
        args: vec![],
    }
}

pub fn encode_heartbeat_probe() -> OscMessage {
    OscMessage {
        addr: "/active".into(),
        args: vec![OscType::Int(0)],
    }
}

fn get_string(args: &[OscType], i: usize) -> Option<String> {
    match args.get(i)? {
        OscType::String(s) => Some(s.clone()),
        _ => None,
    }
}

fn get_f32(args: &[OscType], i: usize) -> Option<f32> {
    match args.get(i)? {
        OscType::Float(f) => Some(*f),
        OscType::Double(d) => Some(*d as f32),
        OscType::Int(v) => Some(*v as f32),
        _ => None,
    }
}

fn get_i32(args: &[OscType], i: usize) -> Option<i32> {
    match args.get(i)? {
        OscType::Int(v) => Some(*v),
        OscType::Float(f) => Some(*f as i32),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rosc::encoder;

    fn roundtrip(msg: OscMessage) -> OscMessage {
        let bytes = encoder::encode(&OscPacket::Message(msg.clone())).unwrap();
        let (_, packet) = rosc::decoder::decode_udp(&bytes).unwrap();
        match packet {
            OscPacket::Message(m) => m,
            _ => panic!("expected a message"),
        }
    }

    #[test]
    fn encode_control_messages_addresses_and_types() {
        assert_eq!(encode_prompt("a cat").addr, "/prompt");
        assert_eq!(
            encode_prompt("a cat").args,
            vec![OscType::String("a cat".into())]
        );
        assert_eq!(encode_negative("blur").addr, "/negative");
        assert_eq!(encode_seed(-1).args, vec![OscType::Int(-1)]);
        assert_eq!(encode_guidance(1.0).args, vec![OscType::Float(1.0)]);
        assert_eq!(encode_delta(1.0).args, vec![OscType::Float(1.0)]);
        assert_eq!(encode_maxfps(30).args, vec![OscType::Int(30)]);
        assert_eq!(
            encode_input("Cam").args,
            vec![OscType::String("Cam".into())]
        );
        assert_eq!(encode_play().args.len(), 0);
        assert_eq!(encode_pause().args.len(), 0);
        assert_eq!(encode_stop().args.len(), 0);
        assert_eq!(encode_unload().args.len(), 0);
    }

    #[test]
    fn steps_multi_argument_roundtrip() {
        let msg = encode_steps(&[17, 23, 31]);
        assert_eq!(msg.args.len(), 3);
        let decoded = roundtrip(msg);
        assert_eq!(
            decoded.args,
            vec![OscType::Int(17), OscType::Int(23), OscType::Int(31)]
        );
    }

    #[test]
    fn decode_telemetry() {
        let active = roundtrip(OscMessage {
            addr: "/active".into(),
            args: vec![OscType::Int(1)],
        });
        assert!(matches!(
            decode(&OscPacket::Message(active)),
            Some(OscIn::Active(1))
        ));

        let fps = roundtrip(OscMessage {
            addr: "/fps".into(),
            args: vec![OscType::Float(23.5)],
        });
        match decode(&OscPacket::Message(fps)) {
            Some(OscIn::Fps(f)) => assert!((f - 23.5).abs() < 1e-6),
            other => panic!("unexpected {other:?}"),
        }

        let out = roundtrip(OscMessage {
            addr: "/output_name".into(),
            args: vec![OscType::String("Prism".into())],
        });
        assert!(
            matches!(decode(&OscPacket::Message(out)), Some(OscIn::OutputName(n)) if n == "Prism")
        );

        let inp = roundtrip(OscMessage {
            addr: "/input_name".into(),
            args: vec![OscType::String("Cam 1".into())],
        });
        assert!(
            matches!(decode(&OscPacket::Message(inp)), Some(OscIn::InputName(n)) if n == "Cam 1")
        );

        let frames = roundtrip(OscMessage {
            addr: "/frames".into(),
            args: vec![OscType::Int(42)],
        });
        assert!(matches!(
            decode(&OscPacket::Message(frames)),
            Some(OscIn::Frames(42))
        ));
    }

    #[test]
    fn decode_error_two_strings() {
        let err = roundtrip(OscMessage {
            addr: "/error".into(),
            args: vec![
                OscType::String("/steps".into()),
                OscType::String("bad count".into()),
            ],
        });
        match decode(&OscPacket::Message(err)) {
            Some(OscIn::Error { address, reason }) => {
                assert_eq!(address, "/steps");
                assert_eq!(reason, "bad count");
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn decode_malformed_error_is_none() {
        // /error with only one argument is malformed and must be dropped.
        let bad = roundtrip(OscMessage {
            addr: "/error".into(),
            args: vec![OscType::String("only".into())],
        });
        assert!(decode(&OscPacket::Message(bad)).is_none());
    }

    #[test]
    fn unknown_address_is_marked_unknown() {
        let msg = roundtrip(OscMessage {
            addr: "/nope".into(),
            args: vec![],
        });
        assert!(matches!(
            decode(&OscPacket::Message(msg)),
            Some(OscIn::Unknown)
        ));
    }

    #[test]
    fn bundles_are_ignored() {
        let bundle = OscPacket::Bundle(rosc::OscBundle {
            timetag: rosc::OscTime {
                seconds: 0,
                fractional: 0,
            },
            content: vec![],
        });
        assert!(decode(&bundle).is_none());
    }
}
