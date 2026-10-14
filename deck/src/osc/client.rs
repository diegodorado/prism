use rosc::encoder;
use std::net::{SocketAddr, UdpSocket};
use std::sync::mpsc;

pub struct OscClient {
    tx: mpsc::Sender<rosc::OscMessage>,
}

impl OscClient {
    /// Start a dedicated send thread on an already-bound socket.
    ///
    /// OSC is stateless over UDP, so a single socket is shared with the
    /// receiver. The thread owns its clone; callers only push messages onto a
    /// channel, which keeps every send off the UI thread.
    pub fn new(socket: UdpSocket, target_addr: SocketAddr, log_tx: mpsc::Sender<String>) -> Self {
        let (tx, rx) = mpsc::channel::<rosc::OscMessage>();
        std::thread::spawn(move || {
            while let Ok(msg) = rx.recv() {
                let packet = rosc::OscPacket::Message(msg.clone());
                if let Ok(buf) = encoder::encode(&packet) {
                    if socket.send_to(&buf, target_addr).is_ok() {
                        let _ = log_tx.send(format!("-> {}", msg.addr));
                    } else {
                        let _ = log_tx.send(format!("-> {} (send failed)", msg.addr));
                    }
                }
            }
        });
        Self { tx }
    }

    pub fn send(&self, msg: rosc::OscMessage) {
        let _ = self.tx.send(msg);
    }
}
