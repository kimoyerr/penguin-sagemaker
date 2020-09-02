using Turing, Distributions
using MCMCChains, Plots, StatsPlots
using NNlib: softmax
using Random
using DataFrames
using ArviZ
using PyPlot
Random.seed!(0)

# Function to convert categorical to one-hot
function onehot(df, sel_cols)
    for col in sel_cols
        categorical!(df, col)
        for l in levels(df[!,col])
            print(l)
            df[!, col * "_" * l] = [r[col] == l ? 1.0 : 0.0 for r in eachrow(df)]
        end
    end
end

# Function to split samples.
function split_data(df, at)
    (r, _) = size(df)
    index = Int(round(r * at))
    train = df[1:index, :]
    test  = df[(index+1):end, :]
    return train, test
end

# Function to rescale variables
function rescale(df, sel_cols)
    for col in sel_cols
        df[!, col] = (df[:,col] .- mean(skipmissing(df[:,col]))) ./ std(skipmissing(df[:,col]))
    end
end

# Import the PenguinDataset
using CSV
data = CSV.read("/home/ubuntu/penguin-sagemaker/data/penguins.csv", missingstring="NA");
describe(data)

# Remove missing rows
dropmissing!(data)

# Change some columns to categorical
onehot(data, ["species", "island", "sex"])
# Rescale the continous features
rescale(data, ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"])

label_names = ["species_Adelie", "species_Gentoo", "species_Chinstrap"]
remove_cols = ["Column1", "species", "island", "sex"]
feature_names = setdiff(names(data), label_names)
feature_names = setdiff(feature_names, remove_cols)

# Randomly shuffle the rows of the dataset
num_rows = size(data, 1)
data = data[Random.shuffle(1:num_rows), :]
first(data,20)

# Split our dataset 75/25 into training/test sets.
train, test = split_data(data, 0.75);
#
# # Create our labels. These are the values we are trying to predict.
train_labels = train[:, label_names]
test_labels = test[:, label_names]
#
# # Create our features. These are our predictors.
train_features = train[:, feature_names];
test_features = test[:, feature_names];

# Convert the DataFrame objects to matrices.
train_labels_mat = Matrix(train_labels);
test_labels_mat = Matrix(test_labels);
sum(train_labels_mat,dims=2)

train_features_mat = Matrix(train_features);
test_features_mat = Matrix(test_features);

# Bayesian multinomial logistic regression
# @model logistic_regression(x, y, n, σ) = begin
#     intercept_Gentoo ~ Normal(0, σ)
#     intercept_Chinstrap ~ Normal(0, σ)
#
#     bill_length_mm_Gentoo ~ Normal(0, σ)
#     bill_length_mm_Chinstrap ~ Normal(0, σ)
#
#     bill_depth_mm_Gentoo ~ Normal(0, σ)
#     bill_depth_mm_Chinstrap ~ Normal(0, σ)
#
#     flipper_length_mm_Gentoo ~ Normal(0, σ)
#     flipper_length_mm_Chinstrap ~ Normal(0, σ)
#
#     body_mass_g_Gentoo ~ Normal(0, σ)
#     body_mass_g_Chinstrap ~ Normal(0, σ)
#
#     island_Biscoe_Gentoo ~ Normal(0, σ)
#     island_Biscoe_Chinstrap ~ Normal(0, σ)
#     island_Dream_Gentoo ~ Normal(0, σ)
#     island_Dream_Chinstrap ~ Normal(0, σ)
#     island_Torgersen_Gentoo ~ Normal(0, σ)
#     island_Torgersen_Chinstrap ~ Normal(0, σ)
#
#     sex_female_Gentoo ~ Normal(0, σ)
#     sex_female_Chinstrap ~ Normal(0, σ)
#     sex_male_Gentoo ~ Normal(0, σ)
#     sex_male_Chinstrap ~ Normal(0, σ)
#
#
#     for i = 1:n
#         v = softmax([0, # this 0 corresponds to the base category `setosa`
#                      intercept_Gentoo + bill_length_mm_Gentoo*x[i, 1] +
#                                             bill_depth_mm_Gentoo*x[i, 2] +
#                                             flipper_length_mm_Gentoo*x[i, 3] +
#                                             body_mass_g_Gentoo*x[i, 4] +
#                                             island_Biscoe_Gentoo*x[i, 5] +
#                                             island_Dream_Gentoo*x[i, 6] +
#                                             island_Torgersen_Gentoo*x[i, 7] +
#                                             sex_female_Gentoo*x[i,8] +
#                                             sex_male_Gentoo*x[i,9],
#                      intercept_Chinstrap + bill_length_mm_Chinstrap*x[i, 1] +
#                                             bill_depth_mm_Chinstrap*x[i, 2] +
#                                             flipper_length_mm_Chinstrap*x[i, 3] +
#                                             body_mass_g_Chinstrap*x[i, 4] +
#                                             island_Biscoe_Chinstrap*x[i, 5] +
#                                             island_Dream_Chinstrap*x[i, 6] +
#                                             island_Torgersen_Chinstrap*x[i, 7] +
#                                             sex_female_Chinstrap*x[i,8] +
#                                             sex_male_Chinstrap*x[i,9]])
#         y[i, :] ~ Multinomial(1, v)
#     end
# end;

# Bayesian multinomial logistic regression
@model logistic_regression(x, y, n, σ) = begin
    intercept_Adelie ~ Normal(0, σ)
    intercept_Gentoo ~ Normal(0, σ)
    intercept_Chinstrap ~ Normal(0, σ)

    bill_length_mm_Adelie ~ Normal(0, σ)
    bill_length_mm_Gentoo ~ Normal(0, σ)
    bill_length_mm_Chinstrap ~ Normal(0, σ)

    bill_depth_mm_Adelie ~ Normal(0, σ)
    bill_depth_mm_Gentoo ~ Normal(0, σ)
    bill_depth_mm_Chinstrap ~ Normal(0, σ)

    flipper_length_mm_Adelie ~ Normal(0, σ)
    flipper_length_mm_Gentoo ~ Normal(0, σ)
    flipper_length_mm_Chinstrap ~ Normal(0, σ)

    body_mass_g_Adelie ~ Normal(0, σ)
    body_mass_g_Gentoo ~ Normal(0, σ)
    body_mass_g_Chinstrap ~ Normal(0, σ)

    island_Biscoe_Adelie ~ Normal(0, σ)
    island_Biscoe_Gentoo ~ Normal(0, σ)
    island_Biscoe_Chinstrap ~ Normal(0, σ)
    island_Dream_Adelie ~ Normal(0, σ)
    island_Dream_Gentoo ~ Normal(0, σ)
    island_Dream_Chinstrap ~ Normal(0, σ)
    island_Torgersen_Adelie ~ Normal(0, σ)
    island_Torgersen_Gentoo ~ Normal(0, σ)
    island_Torgersen_Chinstrap ~ Normal(0, σ)

    sex_female_Adelie ~ Normal(0, σ)
    sex_female_Gentoo ~ Normal(0, σ)
    sex_female_Chinstrap ~ Normal(0, σ)
    sex_male_Adelie ~ Normal(0, σ)
    sex_male_Gentoo ~ Normal(0, σ)
    sex_male_Chinstrap ~ Normal(0, σ)


    for i = 1:n
        v = softmax([intercept_Adelie + bill_length_mm_Adelie*x[i, 1] +
                       bill_depth_mm_Adelie*x[i, 2] +
                       flipper_length_mm_Adelie*x[i, 3] +
                       body_mass_g_Adelie*x[i, 4] +
                       island_Biscoe_Adelie*x[i, 5] +
                       island_Dream_Adelie*x[i, 6] +
                       island_Torgersen_Adelie*x[i, 7] +
                       sex_female_Adelie*x[i,8] +
                       sex_male_Adelie*x[i,9],
                     intercept_Gentoo + bill_length_mm_Gentoo*x[i, 1] +
                        bill_depth_mm_Gentoo*x[i, 2] +
                        flipper_length_mm_Gentoo*x[i, 3] +
                        body_mass_g_Gentoo*x[i, 4] +
                        island_Biscoe_Gentoo*x[i, 5] +
                        island_Dream_Gentoo*x[i, 6] +
                        island_Torgersen_Gentoo*x[i, 7] +
                        sex_female_Gentoo*x[i,8] +
                        sex_male_Gentoo*x[i,9],
                     intercept_Chinstrap + bill_length_mm_Chinstrap*x[i, 1] +
                        bill_depth_mm_Chinstrap*x[i, 2] +
                        flipper_length_mm_Chinstrap*x[i, 3] +
                        body_mass_g_Chinstrap*x[i, 4] +
                        island_Biscoe_Chinstrap*x[i, 5] +
                        island_Dream_Chinstrap*x[i, 6] +
                        island_Torgersen_Chinstrap*x[i, 7] +
                        sex_female_Chinstrap*x[i,8] +
                        sex_male_Chinstrap*x[i,9]])
        y[i, :] ~ Multinomial(1, v)
    end
end;

# Retrieve the number of observations.
n, _ = size(train_features)

# Sample using HMC.
@time chain = mapreduce(c -> sample(logistic_regression(train_features_mat, train_labels_mat, n, 1), HMC(0.05, 10), 1000),
    chainscat,
    1:4)

describe(chain)
plot(chain)

# ArviZ plots
plot_autocorr(chain)
gcf()
plot_trace(chain)
gcf()

corner(chain, [:bill_depth_mm_Chinstrap, :bill_length_mm_Chinstrap, :body_mass_g_Chinstrap, :flipper_length_mm_Chinstrap])

# Predict
function prediction(x::DataFrame, chain)
    # Retrieve the number of rows.
    n, _ = size(x)

    # Get all sample values
    chain_df = DataFrame(chain)

    # Generate a vector to store our predictions.
    v = Vector{String}(undef, n)

    # Get columns for each species
    Adelie_cols = names(chain_df)[vec([occursin(r"Adelie", x) for x in names(chain_df)])]
    Gentoo_cols = names(chain_df)[vec([occursin(r"Gentoo", x) for x in names(chain_df)])]
    Chinstrap_cols = names(chain_df)[vec([occursin(r"Chinstrap", x) for x in names(chain_df)])]

    # Calculate the softmax function for each element in the test set.
    for i = 1:n    # For each data point
        v_tmp = Vector{String}(undef, size(chain_df)[1])
        for j in 1:size(chain_df)[1] # For each sample from chains
            Adelie_terms = chain_df[j,:intercept_Adelie]
            Gentoo_terms = chain_df[j,:intercept_Gentoo]
            Chinstrap_terms = chain_df[j,:intercept_Chinstrap]
            for k in Adelie_cols    # For each column in the dataframe
                if k!="intercept_Adelie"
                    local k_orig = split(k,"_")
                    k_orig = join(k_orig[1:(length(k_orig)-1)],"_")
                    Adelie_terms += chain_df[j,k]*x[i,k_orig]
                end
            end
            for k in Gentoo_cols    # For each column in the dataframe
                if k!="intercept_Gentoo"
                    local k_orig = split(k,"_")
                    k_orig = join(k_orig[1:(length(k_orig)-1)],"_")
                    Gentoo_terms += chain_df[j,k]*x[i,k_orig]
                end
            end
            for k in Chinstrap_cols # For each column in the dataframe
                if k!="intercept_Chinstrap"
                    local k_orig = split(k,"_")
                    k_orig = join(k_orig[1:(length(k_orig)-1)],"_")
                    Chinstrap_terms += chain_df[j,k]*x[i,k_orig]
                end
            end
            num = softmax([Adelie_terms, Gentoo_terms, Chinstrap_terms]) # this 0 corresponds to the base category `Adelie`])
            c = argmax(num) # we pick the class with the highest probability
            if c == 1
                v_tmp[j] = "Adelie"
            elseif c == 2
                v_tmp[j] = "Gentoo"
            else # c == 3
                @assert c == 3
                v_tmp[j] = "Chinstrap"
            end
        end
        u = unique(v_tmp)
        v[i] = findmax(Dict([(i,count(x->x==i,v_tmp)) for i in u]))[2]
    end
    return v
end;

# Make the predictions.
# First on the training set
train_preds = prediction(train_features, chain)
test_preds = prediction(test_features, chain)

# Accuracy
mean(train_preds .== train[!,:species])
mean(test_preds .== test[!,:species])

# Per group
Adelie_rows = test[!, :species] .== "Adelie"
Gentoo_rows = test[!, :species] .== "Gentoo"
Chinstrap_rows = test[!, :species] .== "Chinstrap"

println("Number of Adelie: $(sum(Adelie_rows))")
println("Number of Gentoo: $(sum(Gentoo_rows))")
println("Number of Chinstrap: $(sum(Chinstrap_rows))")

println("Percentage of Adelie predicted correctly: $(mean(test_preds[Adelie_rows] .== test[Adelie_rows, :species]))")
println("Percentage of Gentoo predicted correctly: $(mean(test_preds[Gentoo_rows] .== test[Gentoo_rows, :species]))")
println("Percentage of Chinstrap predicted correctly: $(mean(test_preds[Chinstrap_rows] .== test[Chinstrap_rows, :species]))")
