using ControlnetApp;
using ControlnetApp.Domain;
using ControlnetApp.Pipeline;
using ControlnetApp.Pipeline.OnnxRunners;
using System.Diagnostics;
using Tests;

//DrawElipseTests.DrawTest();
//return;

//var prompt = "a astronaut in space";
//var prompt = "a girl in space";
var prompt = "a girl, undercut hair, apron, amazing body, pronounced feminine feature, busty, ash blonde"; //GOOD
//var prompt = "a cartoon of a man in a green jacket, a character portrait by Thomas Rowlandson, deviantart contest winner, primitivism, playstation 5 screenshot, dutch golden age, official art";
var negativePrompt = "(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation. tattoo";
//var prompt = "1girl, blonde, long dress, dancing, best quality";
//var prompt = "1girl, blonde, long dress, best quality";
//var negativePrompt = "deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, tattoo";
//string? negativePrompt = null;
//var negativePrompt = "monochrome, lowres, bad anatomy, worst quality, low quality";

//string? negativePrompt = null;


prompt = "full-body,frontview,a young female,blonde,in dress,artstation,best quality";
//negativePrompt = "disfigured,bad art,deformed,extra limbs,extra fingers,mutated hands,poorly drawn hands,poorly drawn face,ugly,bad anatomy,extra arms,extra legs,mutated hands,(((ugly face))),(((bad drawed face)))";

//prompt = "full-body,a astronaut in desert";
negativePrompt = "(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation,(ugly face:5), logo, ugly face, bad face";
//negativePrompt = "(deformed, distorted, disfigured:1.3)";

var config = new StableDiffusionConfig()
{
    NumInferenceSteps = 10,
    GuidanceScale = 1, //3
    ControlnetConditionScale = 1f,
    //ControlnetUnConditionScale = 1f,
};

var cpuSessionOptions = SessionOptionsBuilder.Build(ExecutionProvider.Cpu);
var gpuSessionOptions = SessionOptionsBuilder.Build(ExecutionProvider.DML);
//var gpuSessionOptions = SessionOptionsBuilder.Build(ExecutionProvider.Cuda);

var tokenizer = new Tokenizer(cpuSessionOptions);
var textEncoder = new TextEncoder(cpuSessionOptions);
var textPreprocessor = new TextPreprocessor(tokenizer, textEncoder);
var controlnet = new Controlnet(gpuSessionOptions);
var stableDiffusionCn = new StableDiffusionCn(gpuSessionOptions);
var vaeDecoder = new VaeDecoder(gpuSessionOptions);

var sdcnPipeline = new StableDiffusionControlnetPipeline(
    textPreprocessor,
    controlnet,
    stableDiffusionCn,
    vaeDecoder);

var seed = new Random().Next();

// MOCAP FROM JSONS
var mocapResult = MocapResultRepository.GetMocapResult();
int index = 0;
int count = mocapResult.Points.Count;
var stopwatch = new Stopwatch();

foreach (var framePoints in mocapResult.Points)
{
    index++;
    Console.WriteLine($"Process image[{index}/{count}]");
    var image = BodyDrawer.DrawAutobox(
        new SdSize(config.Width, config.Height),
        mocapResult.Info,
        framePoints);

    var nowDate = DateTime.UtcNow.ToString("yyyy-MM-ddTHH-mm-ss");
    image.SaveAsPng($"sketelon_{nowDate}_f_{index}.png");

    stopwatch.Start();
    var outImage = sdcnPipeline.Inference(
        config,
        prompt,
        image,
        negativePrompt: negativePrompt,
        seed: seed,
        isSameLatents: false);
    stopwatch.Stop();
    Console.WriteLine($"Time for image: {stopwatch.Elapsed}, NumInferenceSteps: {config.NumInferenceSteps}");
    stopwatch.Reset();
    outImage.SaveAsPng($"outImage_{nowDate}_f_{index}.png");

    //break;
}

Console.WriteLine($"Frames Count: {count}");
Console.WriteLine($"NumInferenceSteps: {config.NumInferenceSteps}");


// TEST CRUTCH IMAGE
//int imagesCount = 5;
//for (int i = 0; i < imagesCount; i++)
//{
//    Console.WriteLine($"Process image[{i + 1}/{imagesCount}]");
//    var outImage = sdcnPipeline.Inference(
//        config,
//        prompt,
//        InputControlnetCrutch.Get(),
//        negativePrompt: negativePrompt);

//    outImage.SaveAsPng($"outImage_{DateTime.UtcNow.ToString("yyyy-MM-ddTHH-mm-ss")}_i_{i}.png");
//}


Console.WriteLine("All finished");
