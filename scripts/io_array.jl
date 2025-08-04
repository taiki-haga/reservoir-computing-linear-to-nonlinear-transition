# Write a matrix to a plain text file (space-separated)
function write_to_file(filename, data)
    open(filename, "w") do io
        for row in 1:size(data, 1)
            for col in 1:size(data, 2)
                if col > 1
                    write(io, " ")
                end
                write(io, "$(data[row, col])")
            end
            write(io, "\n")
        end
    end
end

# Write a matrix to a CSV file (comma-separated)
function write_to_csv(filename, data)
    open(filename, "w") do io
        for row in 1:size(data, 1)
            for col in 1:size(data, 2)
                if col > 1
                    write(io, ",")
                end
                write(io, "$(data[row, col])")
            end
            write(io, "\n")
        end
    end
end

# Read a space-separated numeric matrix from a text file
function read_from_file(filename)
    data = []
    open(filename, "r") do io
        for line in eachline(io)
            row = parse.(Float64, split(line))
            push!(data, row)
        end
    end
    return hcat(data...)'
end

# Read a comma-separated numeric matrix from a CSV file
function read_from_csv(filename)
    data = []
    open(filename, "r") do io
        for line in eachline(io)
            row = parse.(Float64, split(line, ","))
            push!(data, row)
        end
    end
    return hcat(data...)'
end
