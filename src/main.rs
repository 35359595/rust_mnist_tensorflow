extern crate tensorflow;
extern crate flate2;

use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::result::Result;
use std::path::Path;
use std::process::exit;
use tensorflow::Code;
use tensorflow::Graph;
use tensorflow::ImportGraphDefOptions;
use tensorflow::Session;
use tensorflow::SessionOptions;
use tensorflow::Status;
use tensorflow::StepWithGraph;
use tensorflow::Tensor;
use std::ops::Range;
use flate2::read::GzDecoder;

mod mnist;
use mnist::import_data;

fn main() {
    // Putting the main code in another function serves two purposes:
    // 1. We can use the `?` operator.
    // 2. We can call exit safely, which does not run any destructors.
    let mut images = import_data("train-images-idx3-ubyte.gz");
    let mut labels = import_data("train-labels-idx1-ubyte.gz");

    //image converter tensor by tensor
    let mut buffer = Vec::new();
    let mut letter_container = Vec::new();
    while let Some(symbol) = images.pop() {
        if buffer.len() < 784 {
            buffer.push(symbol);
        } else {
            letter_container.push(buffer);
            buffer = Vec::new();
        }
    }

    //labels pop garbage off
    for i in 0..8 {
        println!("Disposing {}", labels.remove(i));
    }    

    println!("Collected {} numbers!", letter_container.len());
    println!("size of a number: {}", letter_container[0].len());
//    println!("content of first number: {:?}", letter_container[0]);
//    println!("Content of second number: {:?}", letter_container[1]);
    for i in 0..100 {
        println!("{} label: {}", i, labels[i]);
    }
    println!("Labels count: {}", labels.len());
//    for n in 0..28{}

    // exit(match run() {
    //     Ok(_) => 0,
    //     Err(e) => {
    //         println!("{}", e);
    //         1
    //     }
    // })
}

fn run() -> Result<(), Box<Error>> {
    let filename = "examples/addition-model/model.pb"; // z = x + y
    if !Path::new(filename).exists() {
        return Err(Box::new(Status::new_set(Code::NotFound,
                                            &format!("Run 'python addition.py' to generate {} \
                                                      and try again.",
                                                     filename))
            .unwrap()));
    }

    // Create input variables for our addition
    let mut x = Tensor::new(&[1]);
    x[0] = 2i32;
    let mut y = Tensor::new(&[1]);
    y[0] = 40i32;

    // Load the computation graph defined by regression.py.
    let mut graph = Graph::new();
    let mut proto = Vec::new();
    File::open(filename)?.read_to_end(&mut proto)?;
    graph.import_graph_def(&proto, &ImportGraphDefOptions::new())?;
    println!("{:?}", proto);
    let mut session = Session::new(&SessionOptions::new(), &graph)?;

    // Run the Step
    let mut step = StepWithGraph::new();
    step.add_input(&graph.operation_by_name_required("x")?, 0, &x);
    step.add_input(&graph.operation_by_name_required("y")?, 0, &y);
    let z = step.request_output(&graph.operation_by_name_required("z")?, 0);
    session.run(&mut step)?;

    // Check our results.
    let z_res: i32 = step.take_output(z)?[0];
    println!("{:?}", z_res);

    Ok(())
}