require 'torch'
require 'nn'

utf8 = require 'lua-utf8'

require 'LanguageModel'


-- version of sample which passes in a coroutine to mess with 
-- the probability weights


-- try doing the punctuation comparison by index rather than token
-- I think it's a wide character issue


local cmd = torch.CmdLine()
cmd:option('-checkpoint', 'Projects/Musketeers/Musketeers2/Musketeers2_cp_176000.t7')
cmd:option('-vocab', 'Projects/Musketeers/Musketeers.txt')
cmd:option('-notpunct', '’')
--cmd:option('-notpunct', '')
cmd:option('-suppress', '')
cmd:option('-excavate', 0)
cmd:option('-alliterate', '')
cmd:option('-length', 1000)
cmd:option('-start_text', '')
cmd:option('-sample', 1)
cmd:option('-temperature', .3)


local opt = cmd:parse(arg)


local checkpoint = torch.load(opt.checkpoint)
local model = checkpoint.model

local tokens = {}
local punctuation = {}



model:evaluate()

local punctpat = nil
if opt.notpunct ~= '' then
  punctpat = '[' .. opt.notpunct .. ']'
end

for idx, token in pairs(model.idx_to_token) do
  tokens[idx] = 1
  if token:match('%W') then
    if not punctpat or not (utf8.match(token, punctpat)) then 
      punctuation[idx] = 1
    end
  end
end




local wmap = {}
local words = {}

local f = io.open(opt.vocab, "r")
local text = f:read("*all")
local suppressPat = nil
if #opt.suppress > 0 then
  suppressPat = '[' .. opt.suppress .. ']'
end
for w in string.gmatch(text, "%S+") do
  if suppressPat then
    if not w:find(suppressPat) then
      -- wmap[w] = 1
      words[#words + 1] = w
    end
  else
    -- wmap[w] = 1
    words[#words + 1] = w
  end
end


-- for w, _ in pairs(wmap) do
--     words[ #words + 1 ] = w
-- end

print(#words)





function utf8first(s)
  local o2 = utf8.offset(s, 1)
  if o2 == nil then
    return s
  else
    return s:sub(1, o2 - 1)
  end
end







function get_matches(ws)
  local matches = {}
  for p, _ in pairs(punctuation) do
    matches[p] = 1
  end
  if ws then
    for _, w in pairs(ws) do
      if w then
        local idx = model.token_to_idx[utf8first(w)]
        if idx ~= nil then
          matches[idx] = 1
        else
          -- print("unknown first char in '" .. w .. "'")
          -- print("'" .. utf8first(w) .. "'")
        end
      end
    end
  end
  return matches
end




function matches_to_weights(matches)
  local weights = {}
  for idx, _ in pairs(tokens) do
    if matches[idx] ~= nil then
      weights[idx] = 1
    else
      weights[idx] = 0
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
  local v = { "\n", " " }
  for _, w in pairs(ov) do
    local f = utf8first(w)
    if next_char == f then
      local n = w:sub(#f + 1, #w)
      if n then
        v[#v+1] = n
      end
    end
  end
  return v
end


function make_vocab(vocab_gen)
  return coroutine.create(function(prev_char)
    local ok, vocab = coroutine.resume(vocab_gen, nil)
    local current_word = ''
    while ok
      do
        local weights = {}
        local matches = get_matches(vocab)
        local weights = matches_to_weights(matches)
        p = coroutine.yield(weights)
        local next_idx = p[{1,1}]
        local next_char = model.idx_to_token[next_idx]
        if punctuation[next_idx] then
          --print('"' .. current_word .. '" -> "' .. next_char .. '"')
          ok, vocab = coroutine.resume(vocab_gen, current_word)
          current_word = ''
        else
          current_word = current_word .. next_char
          vocab = prune_vocab(vocab, next_char)
          if #vocab < 1 then
            ok, vocab = coroutine.resume(vocab_gen, nil)
          end
        end
      end
      error("Vocabulary exhausted")
  end)
end


local basic_vocab = coroutine.create(function(used_word)
  while true do
    coroutine.yield(init_vocab(words))
  end
end)


-- very simple, doesn't keep track of word locations

local MAX_AHEAD = 500

local excavate_vocab = coroutine.create(function(used_word)
  local used_word = nil
  while #words > 0 do
    --print("excavate_vocab")
    --print(used_word)
    if used_word ~= nil then
      local i = 0
      while #words > 0 and not utf8.match(words[1], '^' .. used_word) do
        table.remove(words, 1)
        i = i + 1
      end
      --print("Skipped " .. tostring(i) .. " words to " .. used_word)
    end
    if #words > 0 then
      --print(#words)
      local lookahead = { unpack(words, 1, MAX_AHEAD) }
      --print(#lookahead)
      used_word = coroutine.yield(init_vocab(lookahead))
    end
  end
  print("Vocabulary finished") 
end)

function make_alliterate(char)
  return coroutine.create(function(prev_char)
    local first_t = {}
    first_t[model.token_to_idx[char]] = 1
    first_t[model.token_to_idx[char:upper()]] = 1
    first_t[model.token_to_idx[' ']] = 1
    local weight_t = matches_to_weights(first_t)
    local weights = weight_t
    while true
      do
        p = coroutine.yield(weights)
        local next_char = model.idx_to_token[p[{1,1}]]
        if next_char:match("%W") then
          weights = weight_t
        else
          weights = {}
        end
      end
  end)
end



local sample = nil

if opt.alliterate ~= '' then
  local mod = make_alliterate(opt.alliterate:sub(1,1))
  sample = model:sample_hacked(opt, mod)
else
  if opt.excavate ~= 0 then
    local tuner = make_vocab(excavate_vocab)
    sample = model:sample_hacked(opt, tuner)
  else
    local tuner = make_vocab(basic_vocab)
  sample = model:sample_hacked(opt, tuner)
  end
end

print(sample)
