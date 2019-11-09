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

local words = { "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india" }


function get_matches(ws)
  local matches = {}
  matches[' '] = 1
  --print("get matches from words", ws)
  if ws then
    for _, w in pairs(ws) do
      matches[w:sub(1,1)] = 1
    end
  end
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

function init_vocab(ws)
  local v = {}
  for i, w in pairs(ws) do
    v[i] = w
  end
  return v
end

function prune_vocab(ov, next_char)
  local v = {}
  for _, w in pairs(ov) do
    if next_char == w:sub(1, 1) then
      v[#v+1] = w:sub(2, #w)
    end
  end
  return v
end


tuner = coroutine.create(function(prev_char)
  local vocab = init_vocab(words)
  while true
    do
      local weights = {}
      local matches = get_matches(vocab)
      local weights = matches_to_weights(matches)
      --print("matches", matches)
      p = coroutine.yield(weights)
      local next_char = model.idx_to_token[p[{1,1}]]
      --print("Next: '" .. next_char .. "'")
      --print("hit return to continue...")
      --io.read()
      if next_char == ' ' then
        vocab = init_vocab(words)
      else
        vocab = prune_vocab(vocab, next_char)
        if #vocab < 1 then
          print("ran out of vocab")
          vocab = init_vocab(words)
        end
      end
    end
end)


model:evaluate()

for idx, token in pairs(model.idx_to_token) do
  tokens[token] = 1
end


local sample = model:sample_hacked(opt, tuner)

print(sample)
