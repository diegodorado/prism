use crate::osc::protocol::{decode, OscIn};
use rosc::decoder;
use std::net::UdpSocket;
use std::sync::mpsc;
use std::time::Duration;

pub struct OscReceiver {
    _handle: std::thread::JoinHandle<()>,
}

impl OscReceiver {
    /// Start a dedicated receive thread on an already-bound socket.
    pub fn new(socket: UdpSocket, event_tx: mpsc::Sender<OscIn>) -> Self {
        socket
            .set_nonblocking(true)
            .expect("failed to set socket to non-blocking");
        let handle = std::thread::spawn(move || {
            let mut buf = [0u8; 65535];
            loop {
                match socket.recv_from(&mut buf) {
                    Ok((n, _)) => {
                        if let Ok((_rest, packet)) = decoder::decode_udp(&buf[..n]) {
                            if let Some(ev) = decode(&packet) {
                                if event_tx.send(ev).is_err() {
                                    break;
                                }
                            }
                        }
                    }
                    Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(2));
                    }
                    Err(_) => {
                        std::thread::sleep(Duration::from_millis(10));
                    }
                }
            }
        });
        Self { _handle: handle }
    }
}
