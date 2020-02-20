#r "../../packages/Angara.Table/lib/net452/Angara.Table.dll"
#r "../../packages/System.Collections.Immutable/lib/portable-net45+win8+wp8+wpa81/System.Collections.Immutable.dll"

open System
open System.IO
open System.Collections.Generic
open Angara.Data

// First generalized reusable code

type Cls = int
type ClassExtractor<'t> = 't -> int

let r = Random(1)

/// Get the K-folds of samples. Each sample can be classified differently (by applying of class extractor)
/// the function will try to reserve the class densities in the training and validation sets
/// Assumptions: there is no dependencies between different classifications
let getKFoldSplit<'sample when 'sample : comparison> (samples:'sample seq) classExtractors foldsCount =
    let samples = Array.ofSeq samples // to iterate once
    let samplesSet = Set.ofSeq samples
    let classExtractors = Array.ofSeq classExtractors
    
    let foldsCountF = float foldsCount

    let Nf = Array.length samples |> float

    let classesCounts =
        classExtractors
        |> Array.map (fun extractor -> Seq.countBy extractor samples |> Map.ofSeq)

    let wholeSetClassesCount =
        classesCounts
        |> Array.map (fun m -> Map.map (fun _ c ->  int(floor (float c)/Nf)))

    let targetValSetClassesCount =
        classesCounts
        |> Array.map (fun m -> Map.map (fun _ c ->  int(floor (float c)/foldsCountF)))

    let rec addOneMoreSampleToValSet validationSamples availableSamples curValSetClassesCount =
        if Set.isEmpty availableSamples then
            validationSamples,curValSetClassesCount
        else
            let ratings =
                curValSetClassesCount
                |> List.mapi (fun idx m -> m |> Map.map (fun k v -> if targetValSetClassesCount.[idx].[k]>v then 1 else 0))
            let sampleRatings =
                availableSamples
                |> Seq.map (fun s -> classExtractors |> Array.mapi (fun idx extr -> ratings.[idx].[extr s]) |> Seq.sum)
            let sampleToAddToValidation =
                Seq.zip availableSamples sampleRatings
                |> Seq.groupBy snd
                |> Seq.sortBy fst
                |> Seq.head |> snd |> Seq.map fst |> Seq.sortBy (fun _ -> r.Next()) |> Seq.head
            let newValSetClassesCount =
                curValSetClassesCount
                |> List.mapi (fun idx m -> let cls = classExtractors.[idx] sampleToAddToValidation in Map.add cls (m.[cls]+1) m)
            addOneMoreSampleToValSet (Set.add sampleToAddToValidation validationSamples) (Set.remove sampleToAddToValidation availableSamples) newValSetClassesCount

    let vaInitialEmptyClassCounters=
        classExtractors
        |> Seq.map (fun extr -> samples |> Seq.map extr |> Seq.distinct |> Seq.map (fun cls -> cls,0) |> Map.ofSeq)
        |> List.ofSeq
    let finalValSamples,resultingValSetClassesCount =
        addOneMoreSampleToValSet Set.empty samplesSet vaInitialEmptyClassCounters
    resultingValSetClassesCount
    |> List.iteri (fun (extIdx:int) (m:Map<int,int>) ->
        printfn "Classification #%d:" extIdx
        m
        |> Map.iter (fun (clsIdx:int) c ->
            printfn "\tclass %3d: %7d samples in va set out of %7d total samples of this class.\tWhole set class portion %.3f%. Val set class portion %.3f%" clsIdx c (classesCounts.[extIdx].[clsIdx]) (wholeSetClassesCount.[extIdx].[clsIdx]*100.0) ((float c)/Nf*100.0)
            )
        )
    finalValSamples
    
//
// Actual script goes below
//

printfn ".NET version %A" Environment.Version
printfn (if Environment.Is64BitProcess then "64-bit" else "32-bit")

let args = Environment.GetCommandLineArgs()

let inputCSV = args.[2]
let outputDir = args.[3]
let folds = System.Int32.Parse(args.[4])

printfn "Will read %s and putting %d fold splits into %s" inputCSV folds outputDir 

type Sample = {
    ID: string
    grapheme_root: int
    vowel_diacritic:int
    consonant_diacritic:int
}

let t = Table<Sample>.Load(inputCSV)
printfn "Loaded info about %d samples" t.RowsCount

let idExtractor (s:Sample) = s.ID
let rootClassExtractor (s:Sample) = s.grapheme_root
let vowelClassExtractpr (s:Sample) = s.vowel_diacritic
let consonant (s:Sample) = s.consonant_diacritic


