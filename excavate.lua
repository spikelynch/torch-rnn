require 'torch'
require 'nn'

require 'LanguageModel'


-- version of sample which passes in a coroutine to mess with 
-- the probability weights

local cmd = torch.CmdLine()
cmd:option('-checkpoint', 'Projects/Musketeers/Musketeers2/Musketeers2_cp_176000.t7')
cmd:option('-length', 5000)
cmd:option('-start_text', '')
cmd:option('-sample', 1)
cmd:option('-temperature', .3)
cmd:option('-verbose', 0)
local opt = cmd:parse(arg)


local checkpoint = torch.load(opt.checkpoint)
local model = checkpoint.model

local tokens = {}

local msg
if opt.verbose == 1 then print(msg) end

local words = { "This", "is", "a", "simple", "set", "of", "words", "to", "match" }


function get_matches(prev, ws)
  local matches = {}
  print("get_matches")
  print(ws)
  for _, w in pairs(ws) do
    print(w)
    wc = w:sub(1,1)
    print("test against: ")
    print(prev)
    if prev then
      if wc == prev then
        matches[wc] = 1
      end
    else
      matches[wc] = 1
    end
  end
  print(matches)
  return matches
end

function matches_to_weights(matches)
  local weights = {}
  for token, _ in pairs(tokens) do
    if matches[token] ~= nil then
      weights[token] = 1
    else
      weights[token] = 0
    end
  end
  return weights
end

tuner = coroutine.create(function(prev_char)
  local prev = nil
  print("start tuner")
  print(words)
  while true
    do
      print("in tuner loop")
      local weights = {}
      local matches = get_matches(prev, words)
      print(matches)
      local weights = matches_to_weights(matches)
      print(weights)
      print(words)
      p = coroutine.yield(weights)
      print(p[{1,1}])
      local prev = model.idx_to_token[p[{1,1}]]
      print(prev)
      print(t)
    end
end)


model:evaluate()

for idx, token in pairs(model.idx_to_token) do
  tokens[token] = 1
end


local sample = model:sample_hacked(opt, tuner)

-- print(sample)
