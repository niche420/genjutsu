use std::sync::Arc;
use std::sync::mpsc::Sender;
use winit::event_loop::EventLoopProxy;
use crate::events::{GenEvent, GjEvent};
use crate::generator::backend::schemas::{JobStatusResponse};

pub struct GenState {
    event_loop_proxy: Arc<EventLoopProxy<GjEvent>>
}

impl GenState {
    pub fn new(event_loop_proxy: Arc<EventLoopProxy<GjEvent>>) -> Self {
        Self {
            event_loop_proxy
        }
    }

    pub fn emit_job_status(&self, id: String, resp: JobStatusResponse) {
        self.event_loop_proxy.send_event(GjEvent::Gen(GenEvent::JobStatus {
            id,
            data: resp.data,
            outputs: resp.outputs
        })).unwrap();
    }
}