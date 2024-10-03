using MLDs
using Documenter

DocMeta.setdocmeta!(MLDs, :DocTestSetup, :(using MLDs); recursive=true)

makedocs(;
    modules=[MLDs],
    authors="Tommaso Stentella <stentell@molgen.mpg.de> and contributors",
    sitename="MLDs.jl",
    format=Documenter.HTML(;
        canonical="https://ArndtLab.github.io/MLDs.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
    warnonly=[:missing_docs],
)

deploydocs(;
    repo="github.com/ArndtLab/MLDs.jl",
    devbranch="main",
)
