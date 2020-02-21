#r "../../packages/Angara.Table/lib/net452/Angara.Table.dll"
#r "../../packages/System.Collections.Immutable/lib/portable-net45+win8+wp8+wpa81/System.Collections.Immutable.dll"

open System
open System.IO
open System.Collections.Generic
open Angara.Data

// First, generalized reusable code

let r = Random(1)

type Cls = int
type ClassExtractor<'t> = 't -> int

/// array - different classifications, Map is classIx -> sample count
type ClsCounters = Map<int,int> array

/// returns the classificationIdx*ClassIdx that has lowest count, but more than 0
let tryFindRarestClass (counters:ClsCounters) = 
    counters
    |> Seq.mapi (fun classificationIdx classification -> Map.toSeq classification |> Seq.map (fun kvp -> classificationIdx,(fst kvp),(snd kvp)))
    |> Seq.concat
    |> Seq.filter (fun x -> let _,_,count = x in count > 0)
    |> Seq.sortBy (fun x -> let _,_,count = x in count)
    |> Seq.tryHead

let createClsCounters clsExtractors samples =
    clsExtractors
        |> Array.map (fun extractor -> Seq.countBy extractor samples |> Map.ofSeq)

let alterClsCounters delta (clsCounters:ClsCounters) classificationIdx classIdx =
    let prevCounter = Map.find classIdx (clsCounters.[classificationIdx])
    clsCounters.[classificationIdx] <- Map.add classIdx (prevCounter+delta) clsCounters.[classificationIdx]

let zeroClsCounters (clsCounters:ClsCounters) =
    clsCounters
    |> Array.map (fun classification -> Map.map (fun _ _ -> 0) classification)

/// Get the K-folds of samples. Each sample can be classified differently (by applying of class extractor)
/// the function will try to reserve the class densities in the training and validation sets
/// Assumptions: there is no dependencies between different classifications
let getKFoldSplit<'sample when 'sample : comparison> (samples:'sample seq) classExtractors foldsCount =
    let samples = Array.ofSeq samples // to iterate once
    let samplesSet = Set.ofSeq samples
    let classExtractors = Array.ofSeq classExtractors
    
    let createClassCounters = createClsCounters classExtractors

    let foldsCountF = float foldsCount

    let Nf = Array.length samples |> float
    let valTagetSize = int(Nf / foldsCountF)

    let wholeSetClassesCounts = createClassCounters samplesSet

    let targetValSetClassesCount =
        wholeSetClassesCounts
        |> Array.map (fun m -> Map.map (fun _ c ->  int(floor (float c)/foldsCountF)) m)
    
    let getValSample trainSetSamples =
        let unassignedSamples = samplesSet - trainSetSamples
        
        let unassignedClsCounters = Array.copy targetValSetClassesCount
        let valClsCounters = zeroClsCounters unassignedClsCounters
    
        let mutable lastReport = DateTime.UtcNow

        let rec getValSampleRec vaSamples unassignedSamples  =
            let valSize = Set.count vaSamples
            let nowTime = DateTime.UtcNow
            if (nowTime - lastReport).TotalMinutes>0.1 then
                printfn "Selected %d samples for val set out of %d needed (%.2f%%). %d are still available for selection" valSize valTagetSize ((float valSize)/(float valTagetSize)*100.0) (Set.count unassignedSamples)
                lastReport <- nowTime
            match tryFindRarestClass unassignedClsCounters with
            |   None ->
                printfn "Done. %d samples selected" valSize
                vaSamples,valClsCounters
            |   Some(rarestClassification,rarestClass,rarestCount) ->
                //printfn "rarest class found %d %d" rarestClassification rarestClass
                let rarestClassUnassignedSamples = 
                    unassignedSamples
                    |> Seq.filter (fun s -> classExtractors.[rarestClassification] s = rarestClass)
                    |> Array.ofSeq
                let nextInValSetOption =
                    rarestClassUnassignedSamples
                    |> Seq.sortBy (fun _ -> r.Next())
                    |> Seq.tryHead
                match nextInValSetOption with
                | Some(nextInValSet) ->
                    // printfn "added sample %A to val set" nextInValSet
                    let counterIdxToAlter =
                        classExtractors
                        |> Seq.mapi (fun idx extractor -> idx,(extractor nextInValSet))
                    counterIdxToAlter
                    |> Seq.iter (fun inc ->
                        let classifIdx,classIdx = inc
                        alterClsCounters 1 valClsCounters classifIdx classIdx
                        alterClsCounters -1 unassignedClsCounters classifIdx classIdx)

                    let unassignedSamples =
                        if rarestCount = 1 then
                            // that was last needed representative of that class
                            // all other unassigned samples holding this class must go to training set
                            // printfn "Needed number of class #%d samples of classification #%d is selected for val set." rarestClass rarestClassification
                            let toTrainingSet =
                                rarestClassUnassignedSamples
                                |> Seq.filter (fun s -> s <> nextInValSet)
                                |> Set.ofSeq
                            unassignedSamples - toTrainingSet
                        else
                            unassignedSamples
                        
                    getValSampleRec (Set.add nextInValSet vaSamples) (Set.remove nextInValSet unassignedSamples)
                | None ->
                    // the class is depleted
                    // the class portion will be biased
                    printfn "Class %d of classification %d will be with biased portion as there are no available samples of this class" rarestClass rarestClassification
                    alterClsCounters (-rarestCount) unassignedClsCounters rarestClassification rarestClass
                    getValSampleRec vaSamples unassignedSamples

        getValSampleRec Set.empty unassignedSamples

    let rec getKFoldRec folds trainSetSamples runsRemain = 
        printfn "starting forming fold %d" (foldsCount - runsRemain + 1)
        if runsRemain = 1 then
            // last fold
            let valSet = samplesSet - trainSetSamples
            (valSet,(createClassCounters valSet),wholeSetClassesCounts) :: folds
        else
            let valSet,valSetCounters = getValSample trainSetSamples
            let valSetSize = Set.count valSet
            printfn "Fold %d (val is %.2f%% of initial size) is ready" valSetSize ((float valSetSize)/Nf*100.0)
            getKFoldRec ((valSet,valSetCounters,wholeSetClassesCounts) :: folds) (trainSetSamples+valSet) (runsRemain - 1)
    
    getKFoldRec [] Set.empty foldsCount
    
//
// Actual script goes below
//

printfn ".NET version %A" Environment.Version
printfn (if Environment.Is64BitProcess then "64-bit" else "32-bit")

let args = Environment.GetCommandLineArgs()

let inputCSV = args.[2]
let outputDir = args.[3]
let foldsCount = System.Int32.Parse(args.[4])

printfn "Will read %s and putting %d fold splits into %s" inputCSV foldsCount outputDir 

type Sample = {
    image_id: string
    grapheme_root: int
    vowel_diacritic:int
    consonant_diacritic:int
}

let readSettings:DelimitedFile.ReadSettings = {
    DelimitedFile.ReadSettings.Default with
        ColumnTypes = Some(fun col -> let idx,name = col in if idx>0 && idx<=3 then Some(typedefof<Int32>) else None)
}

let t = Table<Sample>.Load(inputCSV,readSettings)
printfn "Loaded info about %d samples" t.RowsCount

let samples = t.ToRows<Sample>() |> Array.ofSeq

let idExtractor (s:Sample) = s.image_id
let rootClassExtractor (s:Sample) = s.grapheme_root
let vowelClassExtractor (s:Sample) = s.vowel_diacritic
let consonantClassExtractor (s:Sample) = s.consonant_diacritic

let classExtractors = [| rootClassExtractor; vowelClassExtractor; consonantClassExtractor |]

printfn "Generating folds..."

let folds = getKFoldSplit samples classExtractors foldsCount

printfn "Folds are ready. Dumping to disk"

type Stats = {
    valSamples: int
    trSamples: int
    valPercent: float
}

let dumpFold idx fold =
    let (valSamples:Sample seq),(valCounters:ClsCounters),(totCounters:ClsCounters) = fold
    let valIds = Seq.map idExtractor valSamples
    let valIdsStr = sprintf "image_id\r\n%s" (String.concat "\r\n" valIds)
    let valFileName = Path.Combine(outputDir,sprintf "%d.val_ids.csv" idx)
    File.WriteAllText(valFileName,valIdsStr)

    let statsFolder totalCounterMap stats clsIdx valSamples =
        let totalSamples = Map.find clsIdx totalCounterMap
        let newStat =
            {
                valSamples = valSamples;
                trSamples = totalSamples - valSamples;
                valPercent = (float valSamples)/(float(totalSamples - valSamples))*100.0
            }
        newStat::stats
    
    let dumpStatsFile classificationIdx suffix =
        let rootStats = Map.fold (statsFolder totCounters.[classificationIdx]) [] valCounters.[classificationIdx]
        let rootStatsFile = Path.Combine(outputDir,sprintf "%d.%s.csv" idx suffix)
        Table.Save(Table.OfRows<Stats>(rootStats),rootStatsFile)
    dumpStatsFile 0 "grapheme_root_stats"
    dumpStatsFile 1 "vowel_diacritic_stats"
    dumpStatsFile 2 "consonant_diacritic_stats"

folds |> List.iteri dumpFold

printfn "Done"